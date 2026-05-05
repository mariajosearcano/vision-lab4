"""
Reto: Transformer para restauracion de imagenes
Curso Vision por computador e IA - Guia 4

Este script implementa el reto completo en formato .py:
1) Lee minimo 3 imagenes con defectos/ruido o genera defectos desde ground truth.
2) Aplica SwinIR oficial preentrenado para restauracion/denoising, sin entrenar ni reentrenar.
3) Calcula metricas ENL, SSIM y PSNR contra ground truth.
4) Guarda evidencia visual antes/despues y un resumen CSV con conclusiones automaticas.

Uso recomendado en Colab o entorno local con GPU:
    pip install torch torchvision opencv-python scikit-image matplotlib pandas tqdm pillow requests timm==0.4.12
    python reto_transformer_restauracion_imagenes.py \
        --gt_dir /ruta/ground_truth \
        --output_dir resultados_swinir \
        --generate_defects \
        --max_side 384 \
        --tile 0 \
        --defect_type speckle \
        --noise_level 25

Si ya tienes pares imagen ruidosa / ground truth con el mismo nombre:
    python reto_transformer_restauracion_imagenes.py \
        --input_dir /ruta/imagenes_ruidosas \
        --gt_dir /ruta/ground_truth \
        --output_dir resultados_swinir

Notas:
- Por defecto usa SwinIR color denoising con sigma=25.
- No entrena el modelo. Descarga pesos oficiales si no existen.
- En CPU se recomienda usar copias pequenas, por ejemplo --max_side 384 --tile 0.
- Se basa en el estilo de los notebooks enviados: OpenCV para lectura, NumPy para arreglos,
  Matplotlib para evidencia, metricas numericas y rutas configurables.
"""

### Correr asi: python reto_transformer_restauracion_imagenes.py --gt_dir img/ --output_dir .\resultados_swinir --generate_defects --max_images 3   

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from PIL import Image
except ImportError:  # Pillow suele venir con matplotlib; si no esta, se usa OpenCV normal.
    Image = None
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_COLOR_DN_URLS = {
    15: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    25: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    50: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
}
SWINIR_REQUIRED_PACKAGES = {
    "timm": "timm==0.4.12",
    "requests": "requests",
}


@dataclass
class ImageResult:
    image_name: str
    enl_input: float
    enl_restored: float
    psnr_input: float
    psnr_restored: float
    ssim_input: float
    ssim_restored: float
    analysis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reto Transformer para restauracion de imagenes con SwinIR")
    parser.add_argument("--input_dir", type=str, default=None, help="Carpeta de imagenes degradadas/ruidosas")
    parser.add_argument("--gt_dir", type=str, required=True, help="Carpeta de imagenes ground truth")
    parser.add_argument("--output_dir", type=str, default="resultados_swinir", help="Carpeta de salida")
    parser.add_argument("--swinir_repo", type=str, default="SwinIR", help="Ruta local donde clonar/usar el repo oficial SwinIR")
    parser.add_argument("--model_path", type=str, default=None, help="Ruta opcional a pesos .pth de SwinIR")
    parser.add_argument("--sigma", type=int, default=25, choices=[15, 25, 50], help="Nivel del modelo SwinIR denoising color")
    parser.add_argument("--generate_defects", action="store_true", help="Generar imagenes defectuosas desde gt_dir si no existe input_dir")
    parser.add_argument("--defect_type", type=str, default="speckle", choices=["speckle", "gaussian", "borders", "mixed"], help="Tipo de degradacion sintetica")
    parser.add_argument("--noise_level", type=float, default=25.0, help="Intensidad de ruido sintetico")
    parser.add_argument("--max_images", type=int, default=0, help="Maximo de imagenes a procesar. 0 = todas")
    parser.add_argument("--max_side", type=int, default=384, help="Lado maximo en pixeles para copias de trabajo. 0 = conservar tamano original")
    parser.add_argument("--tile", type=int, default=0, help="Tamano de tile para SwinIR. 0 = procesar la imagen completa; usar multiplos de 8")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Solape entre tiles")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    current_max_side = max(h, w)
    if current_max_side <= max_side:
        return img
    scale = max_side / current_max_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def choose_imread_flag(path: Path, max_side: int) -> int:
    if max_side <= 0 or Image is None:
        return cv2.IMREAD_COLOR
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        return cv2.IMREAD_COLOR

    largest_side = max(w, h)
    if largest_side > max_side * 8:
        return cv2.IMREAD_REDUCED_COLOR_8
    if largest_side > max_side * 4:
        return cv2.IMREAD_REDUCED_COLOR_4
    if largest_side > max_side * 2:
        return cv2.IMREAD_REDUCED_COLOR_2
    return cv2.IMREAD_COLOR


def read_rgb(path: Path, max_side: int = 0) -> np.ndarray:
    img_bgr = cv2.imread(str(path), choose_imread_flag(path, max_side))
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return resize_max_side(img_rgb, max_side)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def prepared_png_path(folder: Path, source_path: Path) -> Path:
    return folder / f"{source_path.stem}.png"


def add_speckle_noise(img: np.ndarray, noise_level: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    sigma = noise_level / 100.0
    noise = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
    noisy = x + x * noise
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.normal(0.0, noise_level, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_border_defects(img: np.ndarray) -> np.ndarray:
    degraded = img.copy()
    h, w = degraded.shape[:2]
    band = max(4, min(h, w) // 18)
    degraded[:band, :] = 0
    degraded[-band:, :] = 0
    degraded[:, :band] = 0
    degraded[:, -band:] = 0
    # Lineas internas simulando defectos de borde o perdida de informacion.
    for _ in range(4):
        y = random.randint(band, max(band, h - band - 1))
        degraded[max(0, y - 1):min(h, y + 2), :] = cv2.GaussianBlur(degraded[max(0, y - 1):min(h, y + 2), :], (7, 7), 0)
    return degraded


def degrade_image(img: np.ndarray, defect_type: str, noise_level: float) -> np.ndarray:
    if defect_type == "speckle":
        return add_speckle_noise(img, noise_level)
    if defect_type == "gaussian":
        return add_gaussian_noise(img, noise_level)
    if defect_type == "borders":
        return add_border_defects(img)
    if defect_type == "mixed":
        return add_border_defects(add_speckle_noise(img, noise_level))
    raise ValueError(f"Tipo de defecto no soportado: {defect_type}")


def prepare_input_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    prepared_gt_dir = output_dir / "ground_truth_preparadas"
    generated_input_dir = output_dir / "imagenes_degradadas_generadas"
    prepared_input_dir = output_dir / "imagenes_degradadas_preparadas"

    if not gt_dir.exists():
        raise FileNotFoundError(f"No existe gt_dir: {gt_dir}")

    gt_images = list_images(gt_dir)
    if len(gt_images) < 3:
        raise ValueError("El reto pide minimo 3 imagenes. gt_dir debe contener al menos 3 imagenes.")
    if args.max_images and args.max_images > 0:
        gt_images = gt_images[: args.max_images]

    max_side = max(0, args.max_side)
    prepared_gt_pairs: List[Tuple[Path, Path]] = []
    if max_side > 0:
        print(f"Preparando copias PNG con lado maximo de {max_side}px...")
        prepared_gt_dir.mkdir(parents=True, exist_ok=True)
        for gt_path in gt_images:
            gt_work_path = prepared_png_path(prepared_gt_dir, gt_path)
            gt_rgb = read_rgb(gt_path, max_side=max_side)
            save_rgb(gt_work_path, gt_rgb)
            prepared_gt_pairs.append((gt_path, gt_work_path))
    else:
        prepared_gt_pairs = [(gt_path, gt_path) for gt_path in gt_images]

    if args.generate_defects or args.input_dir is None:
        generated_input_dir.mkdir(parents=True, exist_ok=True)
        pairs = []
        for original_gt_path, gt_work_path in prepared_gt_pairs:
            img = read_rgb(gt_work_path)
            degraded = degrade_image(img, args.defect_type, args.noise_level)
            input_path = prepared_png_path(generated_input_dir, original_gt_path) if max_side > 0 else generated_input_dir / original_gt_path.name
            save_rgb(input_path, degraded)
            pairs.append((input_path, gt_work_path))
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"No existe input_dir: {input_dir}")
        pairs = []
        if max_side > 0:
            prepared_input_dir.mkdir(parents=True, exist_ok=True)
        for original_gt_path, gt_work_path in prepared_gt_pairs:
            original_input_path = input_dir / original_gt_path.name
            if not original_input_path.exists():
                print(f"Advertencia: no se encontro par degradado para {original_gt_path.name}")
                continue
            if max_side > 0:
                input_rgb = read_rgb(original_input_path, max_side=max_side)
                input_path = prepared_png_path(prepared_input_dir, original_gt_path)
                save_rgb(input_path, input_rgb)
            else:
                input_path = original_input_path
            pairs.append((input_path, gt_work_path))

    if len(pairs) < 3:
        raise ValueError("Se requieren al menos 3 pares imagen degradada / ground truth con el mismo nombre.")
    return pairs


def ensure_swinir_repo(repo_dir: Path) -> None:
    if (repo_dir / "main_test_swinir.py").exists():
        return
    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise FileExistsError(f"{repo_dir} existe, pero no parece ser el repositorio SwinIR oficial.")
    print("Clonando repositorio oficial SwinIR...")
    subprocess.run(["git", "clone", SWINIR_REPO_URL, str(repo_dir)], check=True)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return
    print(f"Descargando pesos: {dest.name}")
    urllib.request.urlretrieve(url, str(dest))


def get_model_path(args: argparse.Namespace, repo_dir: Path) -> Path:
    if args.model_path:
        path = Path(args.model_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe model_path: {path}")
        return path
    model_path = repo_dir / "model_zoo" / "swinir" / Path(SWINIR_COLOR_DN_URLS[args.sigma]).name
    download_file(SWINIR_COLOR_DN_URLS[args.sigma], model_path)
    return model_path


def ensure_swinir_python_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in SWINIR_REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return

    install_cmd = f"python -m pip install {' '.join(missing)}"
    raise SystemExit(
        "Faltan dependencias de SwinIR en este entorno virtual: "
        f"{', '.join(missing)}.\nInstalalas con:\n    {install_cmd}"
    )


def load_swinir_color_denoising_model(repo_dir: Path, model_path: Path, device: torch.device) -> torch.nn.Module:
    repo_path = str(repo_dir.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from models.network_swinir import SwinIR as SwinIRNet

    model = SwinIRNet(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    try:
        pretrained_model = torch.load(str(model_path), map_location=device, weights_only=False)
    except TypeError:
        pretrained_model = torch.load(str(model_path), map_location=device)
    state_dict = pretrained_model["params"] if "params" in pretrained_model else pretrained_model
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def pad_to_window_size(img_lq: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    _, _, h_old, w_old = img_lq.size()
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    if h_pad > 0:
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    if w_pad > 0:
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    return img_lq, h_old, w_old


def tile_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    positions.append(length - tile)
    return positions


def swinir_forward(
    img_lq: torch.Tensor,
    model: torch.nn.Module,
    tile: int,
    tile_overlap: int,
    window_size: int,
    progress_desc: str,
) -> torch.Tensor:
    if tile <= 0:
        return model(img_lq)

    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)
    if tile % window_size != 0:
        raise ValueError(f"--tile debe ser multiplo de {window_size}. Valor recibido: {tile}")
    if tile_overlap < 0 or tile_overlap >= tile:
        raise ValueError("--tile_overlap debe ser mayor o igual a 0 y menor que --tile")

    stride = tile - tile_overlap
    h_idx_list = tile_positions(h, tile, stride)
    w_idx_list = tile_positions(w, tile, stride)
    positions = [(h_idx, w_idx) for h_idx in h_idx_list for w_idx in w_idx_list]

    output_sum = torch.zeros(b, c, h, w).type_as(img_lq)
    output_weight = torch.zeros_like(output_sum)
    for h_idx, w_idx in tqdm(positions, desc=progress_desc, unit="tile", leave=False):
        in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
        out_patch = model(in_patch)
        out_patch_mask = torch.ones_like(out_patch)
        output_sum[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch)
        output_weight[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch_mask)
    return output_sum.div_(output_weight)


def restore_rgb_with_swinir(
    img_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile: int,
    tile_overlap: int,
    progress_desc: str,
) -> np.ndarray:
    window_size = 8
    img_lq = img_rgb.astype(np.float32) / 255.0
    img_lq = np.transpose(img_lq, (2, 0, 1))
    img_lq_tensor = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

    with torch.no_grad():
        img_lq_tensor, h_old, w_old = pad_to_window_size(img_lq_tensor, window_size)
        output = swinir_forward(img_lq_tensor, model, tile, tile_overlap, window_size, progress_desc)
        output = output[..., :h_old, :w_old]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    return (output * 255.0).round().astype(np.uint8)


def estimate_swinir_tiles(img_rgb: np.ndarray, tile: int, tile_overlap: int, window_size: int = 8) -> int:
    if tile <= 0:
        return 1
    h, w = img_rgb.shape[:2]
    h += (window_size - h % window_size) % window_size
    w += (window_size - w % window_size) % window_size
    tile = min(tile, h, w)
    stride = tile - tile_overlap
    if stride <= 0:
        return 0
    return len(tile_positions(h, tile, stride)) * len(tile_positions(w, tile, stride))


def run_swinir_color_denoising(input_dir: Path, restored_dir: Path, repo_dir: Path, model_path: Path, sigma: int, tile: int, tile_overlap: int) -> None:
    restored_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list_images(input_dir)
    if not input_paths:
        raise RuntimeError(f"No se encontraron imagenes para restaurar en {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "Aviso: PyTorch no detecto CUDA; SwinIR correra en CPU. "
            "Para una prueba local rapida usa --max_side 256 o --max_side 384."
        )

    print(f"Cargando SwinIR color denoising sigma={sigma} en {device}...")
    model = load_swinir_color_denoising_model(repo_dir, model_path, device)

    for idx, input_path in enumerate(input_paths, start=1):
        img_rgb = read_rgb(input_path)
        tiles = estimate_swinir_tiles(img_rgb, tile, tile_overlap)
        h, w = img_rgb.shape[:2]
        mode = "imagen completa" if tile <= 0 else f"{tiles} tiles"
        print(f"Restaurando {idx}/{len(input_paths)}: {input_path.name} ({w}x{h}, {mode})")
        restored_rgb = restore_rgb_with_swinir(
            img_rgb=img_rgb,
            model=model,
            device=device,
            tile=tile,
            tile_overlap=tile_overlap,
            progress_desc=f"Tiles {input_path.stem[:24]}",
        )
        save_rgb(restored_dir / f"{input_path.stem}.png", restored_rgb)


def find_restored(restored_dir: Path, original_name: str) -> Optional[Path]:
    original_stem = Path(original_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = restored_dir / f"{original_stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(restored_dir.glob(f"{original_stem}*"))
    matches = [m for m in matches if m.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if matches else None


def calculate_enl(img_rgb: np.ndarray) -> float:
    """Equivalent Number of Looks: media^2 / varianza sobre luminancia.
    Valores mayores suelen indicar regiones mas suavizadas/menos ruido speckle.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = float(np.mean(gray))
    var = float(np.var(gray))
    return (mean * mean) / (var + 1e-8)


def calculate_metrics(gt_rgb: np.ndarray, test_rgb: np.ndarray) -> Tuple[float, float, float]:
    test_rgb = resize_like(test_rgb, gt_rgb)
    psnr = peak_signal_noise_ratio(gt_rgb, test_rgb, data_range=255)
    ssim = structural_similarity(gt_rgb, test_rgb, channel_axis=2, data_range=255)
    enl = calculate_enl(test_rgb)
    return float(enl), float(psnr), float(ssim)


def make_visual_evidence(input_rgb: np.ndarray, restored_rgb: np.ndarray, gt_rgb: np.ndarray, output_path: Path, title: str) -> None:
    restored_rgb = resize_like(restored_rgb, gt_rgb)
    diff_input = cv2.absdiff(gt_rgb, resize_like(input_rgb, gt_rgb))
    diff_restored = cv2.absdiff(gt_rgb, restored_rgb)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        (input_rgb, "Antes: imagen defectuosa"),
        (restored_rgb, "Despues: SwinIR"),
        (gt_rgb, "Ground truth"),
        (diff_input, "Error antes vs GT"),
        (diff_restored, "Error despues vs GT"),
        (cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2GRAY), "Restaurada en gris"),
    ]
    for ax, (img, subtitle) in zip(axes.flat, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(subtitle)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def interpret_result(row: Dict[str, float]) -> str:
    notes = []
    if row["psnr_restored"] > row["psnr_input"]:
        notes.append("mejora PSNR")
    else:
        notes.append("PSNR no mejora")
    if row["ssim_restored"] > row["ssim_input"]:
        notes.append("mejora SSIM")
    else:
        notes.append("SSIM no mejora")
    if row["enl_restored"] > row["enl_input"]:
        notes.append("aumenta ENL: menor textura/ruido aparente")
    else:
        notes.append("ENL no aumenta: conserva textura o mantiene ruido")
    return "; ".join(notes)


def evaluate_all(pairs: List[Tuple[Path, Path]], restored_dir: Path, evidence_dir: Path) -> List[ImageResult]:
    results: List[ImageResult] = []
    for input_path, gt_path in tqdm(pairs, desc="Evaluando metricas"):
        restored_path = find_restored(restored_dir, input_path.name)
        if restored_path is None:
            print(f"Advertencia: no se encontro restauracion para {input_path.name}")
            continue

        input_rgb = read_rgb(input_path)
        gt_rgb = read_rgb(gt_path)
        restored_rgb = read_rgb(restored_path)
        restored_rgb = resize_like(restored_rgb, gt_rgb)
        input_rgb = resize_like(input_rgb, gt_rgb)

        enl_i, psnr_i, ssim_i = calculate_metrics(gt_rgb, input_rgb)
        enl_r, psnr_r, ssim_r = calculate_metrics(gt_rgb, restored_rgb)
        row = {
            "enl_input": enl_i,
            "enl_restored": enl_r,
            "psnr_input": psnr_i,
            "psnr_restored": psnr_r,
            "ssim_input": ssim_i,
            "ssim_restored": ssim_r,
        }
        analysis = interpret_result(row)
        make_visual_evidence(
            input_rgb,
            restored_rgb,
            gt_rgb,
            evidence_dir / f"evidencia_{gt_path.stem}.png",
            title=f"Restauracion con SwinIR - {gt_path.name}",
        )
        results.append(
            ImageResult(
                image_name=gt_path.name,
                enl_input=enl_i,
                enl_restored=enl_r,
                psnr_input=psnr_i,
                psnr_restored=psnr_r,
                ssim_input=ssim_i,
                ssim_restored=ssim_r,
                analysis=analysis,
            )
        )
    return results


def write_csv(results: List[ImageResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imagen",
            "ENL_antes",
            "ENL_despues",
            "PSNR_antes",
            "PSNR_despues",
            "SSIM_antes",
            "SSIM_despues",
            "analisis_visual_y_numerico",
        ])
        for r in results:
            writer.writerow([
                r.image_name,
                f"{r.enl_input:.4f}",
                f"{r.enl_restored:.4f}",
                f"{r.psnr_input:.4f}",
                f"{r.psnr_restored:.4f}",
                f"{r.ssim_input:.4f}",
                f"{r.ssim_restored:.4f}",
                r.analysis,
            ])


def write_conclusions(results: List[ImageResult], output_path: Path) -> None:
    if not results:
        return
    psnr_gain = np.mean([r.psnr_restored - r.psnr_input for r in results])
    ssim_gain = np.mean([r.ssim_restored - r.ssim_input for r in results])
    enl_gain = np.mean([r.enl_restored - r.enl_input for r in results])
    improved_psnr = sum(r.psnr_restored > r.psnr_input for r in results)
    improved_ssim = sum(r.ssim_restored > r.ssim_input for r in results)

    text = f"""# Conclusiones - Reto Transformer para restauracion de imagenes

Imagenes evaluadas: {len(results)}

Promedio de cambio despues de SwinIR:
- Delta PSNR: {psnr_gain:.4f} dB
- Delta SSIM: {ssim_gain:.4f}
- Delta ENL: {enl_gain:.4f}

Interpretacion:
- PSNR mejoro en {improved_psnr}/{len(results)} imagenes.
- SSIM mejoro en {improved_ssim}/{len(results)} imagenes.
- Un aumento de ENL indica suavizado y reduccion aparente de ruido tipo speckle, pero debe revisarse visualmente para confirmar que no se pierdan bordes.
- En las evidencias PNG se recomienda observar: bordes, textura fina, aparicion de artefactos, perdida de detalles y diferencia frente al ground truth.
- Como el reto no exige entrenamiento, se uso un modelo SwinIR preentrenado y solo se realizo inferencia.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    restored_dir = output_dir / "restauradas_swinir"
    evidence_dir = output_dir / "evidencias_visuales"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_swinir_python_dependencies()
    pairs = prepare_input_pairs(args)
    input_dir = pairs[0][0].parent

    repo_dir = Path(args.swinir_repo)
    ensure_swinir_repo(repo_dir)
    model_path = get_model_path(args, repo_dir)

    run_swinir_color_denoising(
        input_dir=input_dir,
        restored_dir=restored_dir,
        repo_dir=repo_dir,
        model_path=model_path,
        sigma=args.sigma,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
    )

    results = evaluate_all(pairs, restored_dir, evidence_dir)
    write_csv(results, output_dir / "metricas_reto_transformer.csv")
    write_conclusions(results, output_dir / "conclusiones_reto_transformer.md")

    print("\nProceso finalizado.")
    print(f"Resultados: {output_dir.resolve()}")
    print(f"CSV metricas: {(output_dir / 'metricas_reto_transformer.csv').resolve()}")
    print(f"Evidencias visuales: {evidence_dir.resolve()}")


if __name__ == "__main__":
    main()
