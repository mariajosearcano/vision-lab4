"""
Reto: Clasificacion de cobertura terrestre (landuse) con aprendizaje profundo
Curso: Vision por computador e IA - Guia practica 4

Este script convierte el flujo de los notebooks base LandUseML.ipynb y
LandUseCNN.ipynb a Python normal (.py), y completa el reto solicitado:
1) Dataset landuse con 3 clases: buildings, airplane, tenniscourt.
2) Modelo base de Machine Learning con caracteristicas RGB estadisticas.
3) CNN secuencial mejorada.
4) Arquitectura no secuencial tipo ResNet ligera.
5) Metricas matematicas y graficas para comparar resultados.
6) Evidencias guardadas en carpeta de salida.

Uso local rapido, desde esta carpeta:
    python reto_landuse_deep_learning.py

Uso completo para entrega/comparacion final:
    python reto_landuse_deep_learning.py --no-fast --deep-models cnn resnet

Uso recomendado en Colab:
    python reto_landuse_deep_learning.py \
        --base-path "/content/drive/MyDrive/2025/Docencia/Visión con IA/4. Aprendizaje Profundo/Ejemplos" \
        --folder landuse \
        --no-fast \
        --deep-models cnn resnet

La estructura esperada es:
base_path/landuse/buildings/*.png
base_path/landuse/airplane/*.png
base_path/landuse/tenniscourt/*.png
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras import Model, layers, regularizers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DEFAULT_CLASSES = ["buildings", "airplane", "tenniscourt"]
IMG_SIZE = 128
SEED = 42


def configure_tensorflow() -> None:
    """Configura GPU si existe. En CPU evita usar mixed precision porque suele ser mas lento."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("TensorFlow: no se detecto GPU. Se entrenara en CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    mixed_precision.set_global_policy("mixed_float16")
    print(f"TensorFlow: GPU detectada ({len(gpus)}). Mixed precision activado.")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_output_dirs(output_dir: str) -> dict:
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


def load_landuse_images(
    base_path: str,
    folder: str,
    classes: list[str],
    img_size: int = IMG_SIZE,
    max_images_per_class: int | None = None,
):
    """Carga imagenes como en LandUseCNN.ipynb: BGR->RGB, resize, normalizacion 0..1."""
    dataset_dir = Path(base_path) / folder
    data, labels, paths = [], [], []

    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")

        image_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
        )
        if max_images_per_class is not None:
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

    if len(data) == 0:
        raise ValueError("No se cargaron imagenes. Revisa --base-path, --folder y las subcarpetas.")

    return np.array(data, dtype="float32"), np.array(labels), np.array(paths)


def extract_rgb_features(images: np.ndarray) -> np.ndarray:
    """Caracteristicas del notebook LandUseML: media y desviacion estandar por canal RGB."""
    features = []
    for img in images:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        features.append([np.mean(r), np.mean(g), np.mean(b), np.std(r), np.std(g), np.std(b)])
    return np.array(features, dtype="float32")


def evaluate_predictions(y_true, y_pred, class_names, model_name, dirs):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

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


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Clase real",
        xlabel="Clase predicha",
        title=f"Matriz de confusion - {title}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_ml_baseline(train_x, test_x, train_y, test_y, class_names, dirs):
    model = RandomForestClassifier(n_estimators=250, random_state=SEED, class_weight="balanced")
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = evaluate_predictions(test_y, pred, class_names, "01_random_forest_ml", dirs)
    return model, metrics


def scaled_filters(base_filters: int, model_scale: float) -> int:
    return max(8, int(round((base_filters * model_scale) / 8)) * 8)


def build_sequential_cnn(input_shape, num_classes, model_scale: float = 1.0):
    """CNN secuencial mejorada a partir de LandUseCNN.ipynb."""
    f1 = scaled_filters(32, model_scale)
    f2 = scaled_filters(64, model_scale)
    f3 = scaled_filters(128, model_scale)
    dense_units = scaled_filters(256, model_scale)

    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(f1, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(f1, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.20),
            layers.Conv2D(f2, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(f2, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(f3, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.50),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ],
        name="cnn_secuencial_mejorada",
    )
    return model


def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_resnet_like(input_shape, num_classes, model_scale: float = 1.0):
    """Arquitectura no secuencial con conexiones residuales tipo ResNet."""
    f1 = scaled_filters(32, model_scale)
    f2 = scaled_filters(64, model_scale)
    f3 = scaled_filters(128, model_scale)
    dense_units = scaled_filters(128, model_scale)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(f1, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, f1, stride=1)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = residual_block(x, f2, stride=1)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = residual_block(x, f3, stride=1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="resnet_ligera_no_secuencial")


def compile_and_train(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    dirs,
    model_name,
    epochs,
    batch_size,
    augment,
    early_stop_patience,
    lr_patience,
):
    opt = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint_path = dirs["models"] / f"{model_name}.keras"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=lr_patience, min_lr=1e-6),
    ]

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.08,
            height_shift_range=0.08,
            zoom_range=0.12,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        history = model.fit(
            datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True),
            validation_data=(test_x, test_y),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            train_x,
            train_y,
            validation_data=(test_x, test_y),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

    plot_history(history, model_name, dirs["plots"] / f"{model_name}_curvas_entrenamiento.png")
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


def evaluate_keras_model(model, test_x, test_y_onehot, class_names, model_name, dirs):
    probs = model.predict(test_x, batch_size=32, verbose=0)
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
            title += f"\n{model_name}: {class_names[pred]}"
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
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(metric_names):
        values = [m[metric] for m in metrics_list]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de metricas: ML vs CNN secuencial vs ResNet no secuencial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "comparacion_metricas.png", dpi=160)
    plt.close(fig)

    with open(dirs["reports"] / "comparacion_metricas.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)


def write_conclusions(metrics_list, dirs, classes):
    sorted_metrics = sorted(metrics_list, key=lambda m: m["accuracy"], reverse=True)
    best = sorted_metrics[0]
    deep_model_names = [m["modelo"] for m in metrics_list if m["modelo"] != "01_random_forest_ml"]
    if deep_model_names:
        comparison_text = (
            "Se comparo un modelo clasico Random Forest basado en medias y desviaciones RGB "
            f"contra {len(deep_model_names)} modelo(s) de aprendizaje profundo: "
            f"{', '.join(deep_model_names)}.\n"
        )
    else:
        comparison_text = "Se entreno el modelo clasico Random Forest basado en medias y desviaciones RGB.\n"

    lines = [
        "CONCLUSIONES DEL RETO LANDUSE\n",
        f"Clases usadas: {', '.join(classes)}.\n",
        comparison_text,
        "Resumen de resultados:\n",
    ]
    for m in metrics_list:
        lines.append(
            f"- {m['modelo']}: accuracy={m['accuracy']:.4f}, "
            f"precision_macro={m['precision_macro']:.4f}, recall_macro={m['recall_macro']:.4f}, "
            f"f1_macro={m['f1_macro']:.4f}\n"
        )
    lines.extend(
        [
            f"\nEl mejor resultado global fue: {best['modelo']} con accuracy={best['accuracy']:.4f}.\n",
            "Analisis: el modelo ML usa solo informacion global de color, por lo que puede fallar "
            "cuando dos coberturas tienen colores parecidos. Las CNN aprovechan patrones espaciales "
            "como formas de edificios, aviones y canchas, por eso normalmente mejoran la clasificacion. "
            "La red no secuencial tipo ResNet facilita el flujo del gradiente mediante conexiones de salto, "
            "lo que puede estabilizar el entrenamiento y mejorar la generalizacion.\n",
            "Para la sustentacion, revisar: matrices de confusion, curvas de loss/accuracy, reporte "
            "por clase y ejemplos visuales de prediccion guardados en la carpeta de evidencias.\n",
        ]
    )
    with open(dirs["reports"] / "conclusiones.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Reto LandUse con aprendizaje profundo")
    parser.add_argument("--base-path", type=str, default=".", help="Ruta base que contiene la carpeta landuse")
    parser.add_argument("--folder", type=str, default="landuse", help="Nombre de la carpeta del dataset")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Tres clases/subcarpetas a usar")
    parser.add_argument("--epochs", type=int, default=30, help="Epocas para entrenar cada red")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Tamano de imagen cuadrada usado por las redes")
    parser.add_argument("--output-dir", type=str, default="evidencias_landuse", help="Carpeta para evidencias")
    parser.add_argument("--test-size", type=float, default=0.25, help="Porcentaje para prueba")
    parser.add_argument(
        "--deep-models",
        nargs="+",
        choices=["cnn", "resnet"],
        default=None,
        help="Redes profundas a entrenar. Por defecto: cnn en modo rapido, cnn resnet con --no-fast.",
    )
    parser.add_argument(
        "--model-scale",
        type=float,
        default=1.0,
        help="Escala de filtros de las redes. 1.0 completo, 0.5 mas rapido.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activa/desactiva aumentacion de datos. En modo rapido se desactiva por defecto.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Paciencia de EarlyStopping")
    parser.add_argument("--lr-patience", type=int, default=4, help="Paciencia de ReduceLROnPlateau")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Limita imagenes por clase para pruebas rapidas. No usar en la corrida final.",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Atajo activado por defecto: menos epocas, imagen mas pequena, batch mayor, red liviana y sin aumentacion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.deep_models is None:
        args.deep_models = ["cnn"] if args.fast else ["cnn", "resnet"]

    if args.fast:
        if args.epochs == 30:
            args.epochs = 8
        if args.img_size == IMG_SIZE:
            args.img_size = 96
        if args.batch_size == 32:
            args.batch_size = 64
        if args.model_scale == 1.0:
            args.model_scale = 0.5
        if args.early_stop_patience == 8:
            args.early_stop_patience = 3
        if args.lr_patience == 4:
            args.lr_patience = 2

    if args.augment is None:
        args.augment = not args.fast

    set_seed(SEED)
    configure_tensorflow()
    dirs = make_output_dirs(args.output_dir)

    if len(args.classes) != 3:
        raise ValueError("El reto pide seleccionar 3 tipos de cobertura. Pasa exactamente 3 clases en --classes.")

    images, string_labels, image_paths = load_landuse_images(
        args.base_path,
        args.folder,
        args.classes,
        args.img_size,
        args.max_images_per_class,
    )

    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(string_labels)
    class_names = list(label_binarizer.classes_)

    if y_onehot.shape[1] == 1:
        raise ValueError("Este script esta preparado para clasificacion multiclase con 3 clases.")

    train_x, test_x, train_y, test_y, train_paths, test_paths = train_test_split(
        images,
        y_onehot,
        image_paths,
        test_size=args.test_size,
        random_state=SEED,
        stratify=string_labels,
    )

    print("\nDistribucion:")
    print(f"Entrenamiento: {train_x.shape}, Prueba: {test_x.shape}")
    print(f"Clases: {class_names}")
    print(
        "Configuracion deep learning: "
        f"modelos={args.deep_models}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"img_size={args.img_size}, model_scale={args.model_scale}, augment={args.augment}"
    )

    # 1) Modelo clasico de ML del notebook LandUseML.ipynb
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(string_labels)
    ml_features = extract_rgb_features(images)
    ml_train_x, ml_test_x, ml_train_y, ml_test_y = train_test_split(
        ml_features,
        y_int,
        test_size=args.test_size,
        random_state=SEED,
        stratify=y_int,
    )
    _, ml_metrics = train_ml_baseline(ml_train_x, ml_test_x, ml_train_y, ml_test_y, list(label_encoder.classes_), dirs)

    # 2) CNN secuencial mejorada
    input_shape = train_x.shape[1:]
    num_classes = len(class_names)
    metrics_list = [ml_metrics]
    probs_by_model = {}

    if "cnn" in args.deep_models:
        sequential_model = build_sequential_cnn(input_shape, num_classes, args.model_scale)
        sequential_model, _ = compile_and_train(
            sequential_model,
            train_x,
            train_y,
            test_x,
            test_y,
            dirs,
            "02_cnn_secuencial_mejorada",
            args.epochs,
            args.batch_size,
            args.augment,
            args.early_stop_patience,
            args.lr_patience,
        )
        seq_probs, seq_metrics = evaluate_keras_model(
            sequential_model,
            test_x,
            test_y,
            class_names,
            "02_cnn_secuencial_mejorada",
            dirs,
        )
        metrics_list.append(seq_metrics)
        probs_by_model["CNN"] = seq_probs

    # 3) Arquitectura no secuencial tipo ResNet
    if "resnet" in args.deep_models:
        resnet_model = build_resnet_like(input_shape, num_classes, args.model_scale)
        resnet_model, _ = compile_and_train(
            resnet_model,
            train_x,
            train_y,
            test_x,
            test_y,
            dirs,
            "03_resnet_ligera_no_secuencial",
            args.epochs,
            args.batch_size,
            args.augment,
            args.early_stop_patience,
            args.lr_patience,
        )
        res_probs, res_metrics = evaluate_keras_model(
            resnet_model,
            test_x,
            test_y,
            class_names,
            "03_resnet_ligera_no_secuencial",
            dirs,
        )
        metrics_list.append(res_metrics)
        probs_by_model["ResNet"] = res_probs

    plot_metrics_comparison(metrics_list, dirs)
    if probs_by_model:
        save_prediction_examples(test_x, test_y, probs_by_model, class_names, dirs)
    write_conclusions(metrics_list, dirs, args.classes)

    print("\nProceso terminado.")
    print(f"Evidencias guardadas en: {Path(args.output_dir).resolve()}")
    print("Archivos clave:")
    print(f"- {dirs['reports'] / 'comparacion_metricas.json'}")
    print(f"- {dirs['reports'] / 'conclusiones.txt'}")
    print(f"- {dirs['plots'] / 'comparacion_metricas.png'}")
    print(f"- {dirs['predictions'] / 'ejemplos_predicciones.png'}")


if __name__ == "__main__":
    main()
