from pathlib import Path
from collections import defaultdict
import csv
import math
import statistics

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline\training\runs\baseline_v1\weights\best.pt"
)

TEST_IMAGES_DIR = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test_dev" / "labels"

BASE_DIR = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "recall_by_input_scale"
    / "analyze_person_recall_by_input_scale_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# PARÁMETROS DE EVALUACIÓN
# ============================================================================

INPUT_SCALES = [640, 960, 1280, 1536]

PERSON_CLASS_ID = 0

CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================================
# INTERVALOS DE TAMAÑO DE PERSONA
# Basados en el área GT en píxeles cuadrados.
# ============================================================================

PERSON_SIZE_BINS = [
    ("<16", 0.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    ("64-128", 64.0, 128.0),
    ("128-256", 128.0, 256.0),
    ("256-512", 256.0, 512.0),
    ("512-1024", 512.0, 1024.0),
    ("1024-2048", 1024.0, 2048.0),
    (">2048", 2048.0, float("inf")),
]


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_recall(tp, gt):
    if gt <= 0:
        return 0.0
    return tp / gt


def get_size_bin(area):
    """
    Devuelve el intervalo de tamaño correspondiente al área.
    """
    for name, low, high in PERSON_SIZE_BINS:
        if low <= area < high:
            return name

    return ">2048"


def calculate_iou(box_a, box_b):
    """
    IoU entre dos cajas en formato:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def xywhn_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    """
    Convierte YOLO normalizado:

        xc yc w h

    a:

        x1 y1 x2 y2
    """

    xc = x_center * image_width
    yc = y_center * image_height

    w = width * image_width
    h = height * image_height

    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0

    return [x1, y1, x2, y2]


def box_area(box):
    """
    Área de una bounding box en píxeles cuadrados.
    """

    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


# ============================================================================
# CARGA DE GROUND TRUTH
# ============================================================================

def load_person_ground_truth(image_path):
    """
    Carga exclusivamente las personas de la etiqueta YOLO correspondiente.

    Devuelve:

    [
        {
            "class_id": 0,
            "box": [x1, y1, x2, y2],
            "area": float,
            "size_bin": str
        },
        ...
    ]
    """

    label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        return []

    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    except Exception:
        return []

    objects = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))

            xc = float(parts[1])
            yc = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            continue

        # Solo PERSON
        if class_id != PERSON_CLASS_ID:
            continue

        box = xywhn_to_xyxy(
            xc,
            yc,
            width,
            height,
            image_width,
            image_height,
        )

        area = box_area(box)

        objects.append(
            {
                "class_id": class_id,
                "box": box,
                "area": area,
                "size_bin": get_size_bin(area),
            }
        )

    return objects


# ============================================================================
# MATCHING GT / PREDICCIONES
# ============================================================================

def match_predictions_to_ground_truth(gt_objects, prediction_boxes):
    """
    Matching greedy por IoU.

    Cada GT puede ser asignado como máximo a una predicción.
    Cada predicción puede utilizarse como máximo una vez.

    Devuelve:

        tp_indices
        fn_indices
        matched_pairs
    """

    if not gt_objects:
        return [], [], []

    if not prediction_boxes:
        return [], list(range(len(gt_objects))), []

    candidates = []

    for gt_index, gt in enumerate(gt_objects):

        gt_box = gt["box"]

        for pred_index, pred in enumerate(prediction_boxes):

            pred_box = pred["box"]

            iou = calculate_iou(gt_box, pred_box)

            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    # Primero los IoU más altos
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pred = set()

    matched_pairs = []

    for iou, gt_index, pred_index in candidates:

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)

        matched_pairs.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": iou,
            }
        )

    tp_indices = sorted(used_gt)

    fn_indices = [
        i
        for i in range(len(gt_objects))
        if i not in used_gt
    ]

    return tp_indices, fn_indices, matched_pairs


# ============================================================================
# PREDICCIONES YOLO
# ============================================================================

def run_inference(model, image_path, input_scale):
    """
    Ejecuta YOLO con el input scale solicitado.

    Devuelve exclusivamente predicciones PERSON.
    """

    try:

        results = model.predict(
            source=str(image_path),
            imgsz=input_scale,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

    except Exception as exc:

        print(
            f"[WARNING] Error procesando "
            f"{image_path.name} con imgsz={input_scale}: {exc}"
        )

        return []

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    predictions = []

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, confidence in zip(
        boxes,
        classes,
        confidences,
    ):

        class_id = int(class_id)

        if class_id != PERSON_CLASS_ID:
            continue

        box_list = [
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
        ]

        predictions.append(
            {
                "class_id": class_id,
                "box": box_list,
                "confidence": float(confidence),
            }
        )

    return predictions


# ============================================================================
# EVALUACIÓN DE UNA ESCALA
# ============================================================================

def evaluate_scale(model, image_paths, input_scale):

    objects = []

    total_gt = 0
    total_tp = 0
    total_fn = 0

    print()
    print("=" * 72)
    print(f"ANALIZANDO INPUT SCALE: {input_scale}")
    print("=" * 72)

    total_images = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):

        gt_objects = load_person_ground_truth(image_path)

        predictions = run_inference(
            model,
            image_path,
            input_scale,
        )

        tp_indices, fn_indices, matched_pairs = (
            match_predictions_to_ground_truth(
                gt_objects,
                predictions,
            )
        )

        tp_set = set(tp_indices)
        fn_set = set(fn_indices)

        for gt_index, gt in enumerate(gt_objects):

            if gt_index in tp_set:
                status = "TP"
            elif gt_index in fn_set:
                status = "FN"
            else:
                status = "FN"

            matched_iou = None

            for match in matched_pairs:

                if match["gt_index"] == gt_index:

                    matched_iou = match["iou"]
                    break

            objects.append(
                {
                    "image": image_path.name,
                    "input_scale": input_scale,
                    "class": "person",
                    "class_id": PERSON_CLASS_ID,
                    "area": float(gt["area"]),
                    "size_bin": gt["size_bin"],
                    "status": status,
                    "iou": (
                        float(matched_iou)
                        if matched_iou is not None
                        else ""
                    ),
                }
            )

        gt_count = len(gt_objects)
        tp_count = len(tp_indices)
        fn_count = len(fn_indices)

        total_gt += gt_count
        total_tp += tp_count
        total_fn += fn_count

        if index % 100 == 0 or index == total_images:

            print(
                f"Analizadas: {index:,}/{total_images:,}"
            )

    recall = safe_recall(
        total_tp,
        total_gt,
    )

    print()
    print(
        f"INPUT SCALE {input_scale}: "
        f"GT={total_gt:,} "
        f"TP={total_tp:,} "
        f"FN={total_fn:,} "
        f"Recall={recall:.4f}"
    )

    return {
        "input_scale": input_scale,
        "person_gt": total_gt,
        "person_tp": total_tp,
        "person_fn": total_fn,
        "person_recall": recall,
        "objects": objects,
    }


# ============================================================================
# ESTADÍSTICAS POR ESCALA
# ============================================================================

def get_scale_statistics(objects, evaluation_results):
    """
    Genera estadísticas robustas.

    IMPORTANTE:
    Esta función NO espera claves externas como person_gt.
    Las calcula directamente a partir de los objetos y de
    evaluation_results.
    """

    rows = []

    for result in evaluation_results:

        scale = result["input_scale"]

        scale_objects = [
            obj
            for obj in objects
            if obj["input_scale"] == scale
        ]

        gt_count = len(scale_objects)

        tp_count = sum(
            1
            for obj in scale_objects
            if obj["status"] == "TP"
        )

        fn_count = sum(
            1
            for obj in scale_objects
            if obj["status"] == "FN"
        )

        areas = [
            float(obj["area"])
            for obj in scale_objects
            if obj.get("area") is not None
        ]

        recall = safe_recall(
            tp_count,
            gt_count,
        )

        if areas:

            mean_area = statistics.mean(areas)
            median_area = statistics.median(areas)
            min_area = min(areas)
            max_area = max(areas)

        else:

            mean_area = 0.0
            median_area = 0.0
            min_area = 0.0
            max_area = 0.0

        size_distribution = defaultdict(int)

        for obj in scale_objects:

            size_distribution[
                obj["size_bin"]
            ] += 1

        row = {
            "input_scale": scale,
            "person_gt": gt_count,
            "person_tp": tp_count,
            "person_fn": fn_count,
            "person_recall": recall,
            "mean_person_area_px2": mean_area,
            "median_person_area_px2": median_area,
            "min_person_area_px2": min_area,
            "max_person_area_px2": max_area,
        }

        for size_name, _, _ in PERSON_SIZE_BINS:

            key = (
                "gt_"
                + size_name
                .replace("<", "lt_")
                .replace(">", "gt_")
                .replace("-", "_")
            )

            row[key] = size_distribution.get(
                size_name,
                0,
            )

        rows.append(row)

    return rows


# ============================================================================
# CSV OBJECTOS
# ============================================================================

def write_objects_csv(objects):

    output_path = (
        REPORTS_DIR
        / "person_recall_by_input_scale_objects_v1.csv"
    )

    fieldnames = [
        "image",
        "input_scale",
        "class",
        "class_id",
        "area",
        "size_bin",
        "status",
        "iou",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for obj in objects:

            writer.writerow(
                {
                    field: obj.get(field, "")
                    for field in fieldnames
                }
            )

    print(f"[OK] {output_path}")

    return output_path


# ============================================================================
# CSV RESUMEN POR INPUT SCALE
# ============================================================================

def write_scale_summary_csv(evaluation_results):

    output_path = (
        REPORTS_DIR
        / "person_recall_by_input_scale_v1.csv"
    )

    fieldnames = [
        "input_scale",
        "person_gt",
        "person_tp",
        "person_fn",
        "person_recall",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in evaluation_results:

            writer.writerow(
                {
                    "input_scale": result["input_scale"],
                    "person_gt": result["person_gt"],
                    "person_tp": result["person_tp"],
                    "person_fn": result["person_fn"],
                    "person_recall": (
                        f"{result['person_recall']:.6f}"
                    ),
                }
            )

    print(f"[OK] {output_path}")

    return output_path


# ============================================================================
# CSV ESTADÍSTICAS
# ============================================================================

def write_statistics_csv(statistics_rows):

    output_path = (
        REPORTS_DIR
        / "input_scale_statistics_v1.csv"
    )

    if not statistics_rows:

        print(
            "[WARNING] No hay estadísticas para escribir."
        )

        return output_path

    # Construimos las columnas directamente desde las filas.
    fieldnames = list(
        statistics_rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in statistics_rows:

            # IMPORTANTE:
            # Nunca asumimos que las claves existen fuera
            # de esta estructura.

            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )

    print(f"[OK] {output_path}")

    return output_path


# ============================================================================
# SUMMARY TXT
# ============================================================================

def write_summary(
    image_count,
    evaluation_results,
):

    output_path = (
        REPORTS_DIR
        / "PERSON_RECALL_BY_INPUT_SCALE_V1_SUMMARY.txt"
    )

    total_gt = 0
    total_tp = 0
    total_fn = 0

    for result in evaluation_results:

        total_gt += result["person_gt"]
        total_tp += result["person_tp"]
        total_fn += result["person_fn"]

    overall_recall = safe_recall(
        total_tp,
        total_gt,
    )

    lines = []

    lines.append("=" * 72)
    lines.append(
        "SAR YOLO26 - PERSON RECALL BY INPUT SCALE ANALYSIS V1"
    )
    lines.append("=" * 72)
    lines.append("")

    lines.append(
        f"Images:              {image_count:,}"
    )

    lines.append(
        f"PERSON GT:           {total_gt:,}"
    )

    lines.append(
        f"PERSON TP:           {total_tp:,}"
    )

    lines.append(
        f"PERSON FN:           {total_fn:,}"
    )

    lines.append(
        f"PERSON Recall:       {overall_recall:.4f}"
    )

    lines.append("")
    lines.append(
        "RECALL POR INPUT SCALE"
    )
    lines.append("")

    for result in evaluation_results:

        lines.append(
            f"{result['input_scale']:>4} "
            f"GT={result['person_gt']:>7,} "
            f"TP={result['person_tp']:>7,} "
            f"FN={result['person_fn']:>7,} "
            f"Recall={result['person_recall']:.4f}"
        )

    lines.append("")
    lines.append(
        "INTERPRETACIÓN"
    )
    lines.append("")

    best_result = max(
        evaluation_results,
        key=lambda x: x["person_recall"],
    )

    baseline_result = next(
        (
            r
            for r in evaluation_results
            if r["input_scale"] == 640
        ),
        None,
    )

    lines.append(
        f"Mejor recall observado: "
        f"input scale {best_result['input_scale']} "
        f"con Recall={best_result['person_recall']:.4f}."
    )

    if baseline_result is not None:

        improvement = (
            best_result["person_recall"]
            - baseline_result["person_recall"]
        )

        lines.append(
            f"Mejora respecto a 640: "
            f"{improvement:+.4f}."
        )

    lines.append("")
    lines.append(
        "NOTA: Las métricas se calculan sobre el mismo conjunto "
        "test_dev y el dataset NO ha sido modificado."
    )

    lines.append(
        "NOTA: El cambio corresponde exclusivamente al input scale "
        "utilizado durante la inferencia."
    )

    lines.append("")

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(f"[OK] {output_path}")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON RECALL BY INPUT SCALE ANALYSIS V1"
    )
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET_ROOT)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    print("Test:")
    print(TEST_IMAGES_DIR)
    print()

    print("Output:")
    print(OUTPUT_DIR)
    print()

    # ------------------------------------------------------------------------
    # VALIDACIONES
    # ------------------------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"No existe DATASET_ROOT:\n{DATASET_ROOT}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES_DIR.exists():

        raise FileNotFoundError(
            f"No existe el directorio de imágenes:\n"
            f"{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():

        raise FileNotFoundError(
            f"No existe el directorio de labels:\n"
            f"{TEST_LABELS_DIR}"
        )

    # ------------------------------------------------------------------------
    # CREAR OUTPUT
    # ------------------------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # IMÁGENES
    # ------------------------------------------------------------------------

    image_paths = sorted(
        [
            path
            for path in TEST_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    print(
        f"Imágenes encontradas: {len(image_paths):,}"
    )

    if not image_paths:

        raise RuntimeError(
            "No se encontraron imágenes en test_dev/images."
        )

    # ------------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------------

    print()
    print("Cargando modelo YOLO26s...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("[OK] Modelo cargado.")
    print()

    # ------------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------------

    all_objects = []
    evaluation_results = []

    for input_scale in INPUT_SCALES:

        result = evaluate_scale(
            model=model,
            image_paths=image_paths,
            input_scale=input_scale,
        )

        evaluation_results.append(
            {
                "input_scale": result["input_scale"],
                "person_gt": result["person_gt"],
                "person_tp": result["person_tp"],
                "person_fn": result["person_fn"],
                "person_recall": result["person_recall"],
            }
        )

        all_objects.extend(
            result["objects"]
        )

    # ------------------------------------------------------------------------
    # CSV 1 - OBJETOS
    # ------------------------------------------------------------------------

    write_objects_csv(
        all_objects
    )

    # ------------------------------------------------------------------------
    # CSV 2 - RESUMEN INPUT SCALE
    # ------------------------------------------------------------------------

    write_scale_summary_csv(
        evaluation_results
    )

    # ------------------------------------------------------------------------
    # ESTADÍSTICAS
    # ------------------------------------------------------------------------

    statistics_rows = get_scale_statistics(
        all_objects,
        evaluation_results,
    )

    # ------------------------------------------------------------------------
    # CSV 3 - ESTADÍSTICAS
    # ------------------------------------------------------------------------

    write_statistics_csv(
        statistics_rows
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    write_summary(
        image_count=len(image_paths),
        evaluation_results=evaluation_results,
    )

    # ------------------------------------------------------------------------
    # RESULTADO FINAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON RECALL BY INPUT SCALE V1"
    )
    print("=" * 72)
    print()

    total_gt = sum(
        r["person_gt"]
        for r in evaluation_results
    )

    print(
        f"Imágenes:              {len(image_paths):,}"
    )

    print(
        f"PERSON GT:             {total_gt:,}"
    )

    print()

    for result in evaluation_results:

        print(
            f"INPUT SCALE "
            f"{result['input_scale']}: "
            f"GT={result['person_gt']:,} "
            f"TP={result['person_tp']:,} "
            f"FN={result['person_fn']:,} "
            f"Recall={result['person_recall']:.4f}"
        )

    print()
    print(
        "[OK] Reports generados."
    )

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    print()
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()