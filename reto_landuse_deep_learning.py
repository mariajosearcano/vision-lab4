"""
Reto: Clasificacion de cobertura terrestre landuse con aprendizaje profundo
Version rapida y con mejor precision.

Idea principal:
- En vez de entrenar CNN/ResNet desde cero, usa Transfer Learning con MobileNetV2 preentrenada en ImageNet.
- Esto suele ser mucho mas rapido y preciso con datasets pequenos.
- Mantiene comparacion contra ML clasico y contra una CNN secuencial ligera.
- Incluye arquitectura no secuencial: MobileNetV2/ResNet-like funcional con transfer learning.

Ejemplo Colab:
python reto_landuse_rapido_y_preciso.py \
  --base-path "/content/drive/MyDrive/2025/Docencia/Visión con IA/4. Aprendizaje Profundo/Ejemplos" \
  --folder landuse \
  --epochs-cnn 8 \
  --epochs-transfer 8 \
  --fine-tune-epochs 3 \
  --img-size 160 \
  --batch-size 32 \
  --max-images-per-class 250

Mas rapido para pruebas:
python reto_landuse_rapido_y_preciso.py --base-path "TU_RUTA" --folder landuse --epochs-cnn 5 --epochs-transfer 5 --fine-tune-epochs 2 --img-size 128 --max-images-per-class 120
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

DEFAULT_CLASSES = ["buildings", "airplane", "tenniscourt"]
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_speed():
    # Activa opciones que ayudan en GPU. Si no hay GPU, no hace dano.
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("GPU detectada: mixed precision activado.")
    except Exception as e:
        print(f"No se pudo activar optimizacion GPU: {e}")


def make_output_dirs(output_dir):
    root = Path(output_dir)
    dirs = {
        "root": root,
        "models": root / "modelos",
        "plots": root / "graficas",
        "predictions": root / "predicciones",
        "reports": root / "reportes",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_landuse_images(base_path, folder, classes, img_size, max_images_per_class=None):
    dataset_dir = Path(base_path) / folder
    data, labels, paths = [], [], []

    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")

        image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
        random.Random(SEED).shuffle(image_files)
        if max_images_per_class is not None and max_images_per_class > 0:
            image_files = image_files[:max_images_per_class]

        print(f"Cargando {class_name}: {len(image_files)} imagenes")
        for path in image_files:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Advertencia: no se pudo leer {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            data.append(img)
            labels.append(class_name)
            paths.append(str(path))

    if not data:
        raise ValueError("No se cargaron imagenes. Revisa --base-path, --folder y clases.")
    return np.array(data, dtype="float32"), np.array(labels), np.array(paths)


def make_tf_dataset(x, y, batch_size, training=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=len(x), seed=SEED, reshuffle_each_iteration=True)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06, fill_mode="reflect"),
            layers.RandomZoom(0.08, fill_mode="reflect"),
            layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
            layers.RandomContrast(0.12),
        ])
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def extract_rgb_features(images):
    features = []
    for img in images:
        features.append([
            np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2]),
            np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2]),
        ])
    return np.array(features, dtype="float32")


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, ylabel="Clase real", xlabel="Clase predicha", title=f"Matriz de confusion - {title}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_predictions(y_true, y_pred, class_names, model_name, dirs):
    report_txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "modelo": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report_dict,
    }
    with open(dirs["reports"] / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
        f.write("\n\nMetricas resumen:\n")
        for k, v in metrics.items():
            if k != "classification_report":
                f.write(f"{k}: {v}\n")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, model_name, dirs["plots"] / f"{model_name}_matriz_confusion.png")
    return metrics


def train_ml_baseline(train_x, test_x, train_y, test_y, class_names, dirs):
    model = RandomForestClassifier(n_estimators=180, random_state=SEED, class_weight="balanced", n_jobs=-1)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = evaluate_predictions(test_y, pred, class_names, "01_random_forest_ml", dirs)
    return model, metrics


def build_fast_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ], name="cnn_secuencial_rapida")
    return model


def build_mobilenet_transfer(input_shape, num_classes):
    # Arquitectura no secuencial/funcional con red preentrenada.
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda z: tf.keras.applications.mobilenet_v2.preprocess_input(z * 255.0))(inputs)
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = Model(inputs, outputs, name="02_mobilenetv2_transfer_no_secuencial")
    return model, base


def compile_and_train(model, train_ds, val_ds, dirs, model_name, epochs, class_weight=None, lr=1e-3):
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint_path = dirs["models"] / f"{model_name}.keras"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=2, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, class_weight=class_weight, verbose=1)
    plot_history(history, model_name, dirs["plots"] / f"{model_name}_curvas.png")
    return model, history


def fine_tune_mobilenet(model, base_model, train_ds, val_ds, dirs, class_weight, epochs=3):
    if epochs <= 0:
        return model, None
    base_model.trainable = True
    # Solo descongelar las ultimas capas para no volver lento el entrenamiento.
    for layer in base_model.layers[:-35]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        ModelCheckpoint(str(dirs["models"] / "03_mobilenetv2_finetuned.keras"), monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True, mode="max"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, class_weight=class_weight, verbose=1)
    plot_history(history, "03_mobilenetv2_finetuned", dirs["plots"] / "03_mobilenetv2_finetuned_curvas.png")
    return model, history


def plot_history(history, model_name, output_path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist["accuracy"], label="Entrenamiento")
    ax.plot(epochs, hist["val_accuracy"], label="Validacion")
    ax.set_title(f"Accuracy - {model_name}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_accuracy.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist["loss"], label="Entrenamiento")
    ax.plot(epochs, hist["val_loss"], label="Validacion")
    ax.set_title(f"Loss - {model_name}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_loss.png"), dpi=160)
    plt.close(fig)


def evaluate_keras_model(model, test_x, test_y_onehot, class_names, model_name, dirs, batch_size):
    probs = model.predict(test_x, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(test_y_onehot, axis=1)
    metrics = evaluate_predictions(y_true, y_pred, class_names, model_name, dirs)
    return probs, metrics


def save_prediction_examples(test_x, test_y, probs_by_model, class_names, dirs, max_examples=12):
    y_true = np.argmax(test_y, axis=1)
    n = min(max_examples, len(test_x))
    indices = np.linspace(0, len(test_x) - 1, n, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    for ax, idx in zip(axes, indices):
        ax.imshow(test_x[idx])
        title = f"Real: {class_names[y_true[idx]]}"
        for model_name, probs in probs_by_model.items():
            pred = np.argmax(probs[idx])
            conf = np.max(probs[idx])
            title += f"\n{model_name}: {class_names[pred]} ({conf:.2f})"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dirs["predictions"] / "ejemplos_predicciones.png", dpi=160)
    plt.close(fig)


def plot_metrics_comparison(metrics_list, dirs):
    model_names = [m["modelo"] for m in metrics_list]
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(model_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        values = [m[metric] for m in metrics_list]
        ax.bar(x + i * width, values, width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de metricas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "comparacion_metricas.png", dpi=160)
    plt.close(fig)
    with open(dirs["reports"] / "comparacion_metricas.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)


def write_conclusions(metrics_list, dirs, classes):
    best = sorted(metrics_list, key=lambda m: m["accuracy"], reverse=True)[0]
    lines = [
        "CONCLUSIONES DEL RETO LANDUSE\n\n",
        f"Clases usadas: {', '.join(classes)}.\n",
        "Esta version usa Transfer Learning con MobileNetV2 para mejorar precision sin entrenar desde cero. ",
        "La CNN secuencial se mantiene como comparacion, pero es mas liviana para reducir tiempo.\n\n",
        "Resumen de resultados:\n",
    ]
    for m in metrics_list:
        lines.append(f"- {m['modelo']}: accuracy={m['accuracy']:.4f}, precision_macro={m['precision_macro']:.4f}, recall_macro={m['recall_macro']:.4f}, f1_macro={m['f1_macro']:.4f}\n")
    lines += [
        f"\nMejor modelo: {best['modelo']} con accuracy={best['accuracy']:.4f}.\n",
        "Analisis: entrenar una ResNet pequena desde cero puede ser rapido, pero con pocas imagenes tiende a sesgarse hacia una clase. ",
        "MobileNetV2 ya trae filtros visuales aprendidos, por eso necesita menos epocas y suele generalizar mejor.\n",
    ]
    with open(dirs["reports"] / "conclusiones.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Reto LandUse rapido y preciso con Transfer Learning")
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--folder", type=str, default="landuse")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--max-images-per-class", type=int, default=250)
    parser.add_argument("--epochs-cnn", type=int, default=8)
    parser.add_argument("--epochs-transfer", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="evidencias_landuse")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--skip-cnn", action="store_true", help="Entrena solo Random Forest y MobileNetV2 para ir mas rapido")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    configure_speed()
    dirs = make_output_dirs(args.output_dir)

    if len(args.classes) != 3:
        raise ValueError("El reto pide exactamente 3 clases.")

    images, string_labels, image_paths = load_landuse_images(args.base_path, args.folder, args.classes, args.img_size, args.max_images_per_class)
    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(string_labels)
    class_names = list(label_binarizer.classes_)

    train_x, test_x, train_y, test_y, train_labels_str, test_labels_str = train_test_split(
        images, y_onehot, string_labels, test_size=args.test_size, random_state=SEED, stratify=string_labels
    )

    print("\nDistribucion:")
    print(f"Entrenamiento: {train_x.shape}, Prueba: {test_x.shape}")
    print(f"Clases: {class_names}")

    y_train_int = np.argmax(train_y, axis=1)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y_train_int)
    class_weight = {i: float(w) for i, w in enumerate(class_weights_array)}
    print(f"Class weights: {class_weight}")

    train_ds = make_tf_dataset(train_x, train_y, args.batch_size, training=True, augment=True)
    val_ds = make_tf_dataset(test_x, test_y, args.batch_size, training=False, augment=False)

    # 1. ML clasico
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(string_labels)
    ml_features = extract_rgb_features(images)
    ml_train_x, ml_test_x, ml_train_y, ml_test_y = train_test_split(ml_features, y_int, test_size=args.test_size, random_state=SEED, stratify=y_int)
    _, ml_metrics = train_ml_baseline(ml_train_x, ml_test_x, ml_train_y, ml_test_y, list(label_encoder.classes_), dirs)

    metrics_list = [ml_metrics]
    probs_by_model = {}

    # 2. CNN secuencial rapida opcional
    if not args.skip_cnn:
        cnn = build_fast_cnn(train_x.shape[1:], len(class_names))
        cnn, _ = compile_and_train(cnn, train_ds, val_ds, dirs, "02_cnn_secuencial_rapida", args.epochs_cnn, class_weight, lr=1e-3)
        cnn_probs, cnn_metrics = evaluate_keras_model(cnn, test_x, test_y, class_names, "02_cnn_secuencial_rapida", dirs, args.batch_size)
        metrics_list.append(cnn_metrics)
        probs_by_model["CNN"] = cnn_probs

    # 3. Transfer learning no secuencial: mucho mas recomendado
    transfer, base = build_mobilenet_transfer(train_x.shape[1:], len(class_names))
    transfer, _ = compile_and_train(transfer, train_ds, val_ds, dirs, "03_mobilenetv2_transfer_no_secuencial", args.epochs_transfer, class_weight, lr=1e-3)
    transfer, _ = fine_tune_mobilenet(transfer, base, train_ds, val_ds, dirs, class_weight, epochs=args.fine_tune_epochs)
    tr_probs, tr_metrics = evaluate_keras_model(transfer, test_x, test_y, class_names, "03_mobilenetv2_transfer_finetuned", dirs, args.batch_size)
    metrics_list.append(tr_metrics)
    probs_by_model["MobileNet"] = tr_probs

    plot_metrics_comparison(metrics_list, dirs)
    save_prediction_examples(test_x, test_y, probs_by_model, class_names, dirs)
    write_conclusions(metrics_list, dirs, args.classes)

    print("\nProceso terminado.")
    print(f"Evidencias guardadas en: {Path(args.output_dir).resolve()}")
    print("Revisa:")
    print(f"- {dirs['reports'] / 'comparacion_metricas.json'}")
    print(f"- {dirs['reports'] / 'conclusiones.txt'}")
    print(f"- {dirs['plots'] / 'comparacion_metricas.png'}")
    print(f"- {dirs['predictions'] / 'ejemplos_predicciones.png'}")


if __name__ == "__main__":
    main()
