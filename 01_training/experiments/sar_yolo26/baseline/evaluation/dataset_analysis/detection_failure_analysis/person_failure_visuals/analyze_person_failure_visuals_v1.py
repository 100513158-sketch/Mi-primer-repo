from pathlib import Path
import csv
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# SAR YOLO26 - PERSON FAILURE VISUALS ANALYSIS V1
# ============================================================

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

BASELINE_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = (
    BASELINE_ROOT
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)

TEST_IMAGES = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS = DATASET_ROOT / "test_dev" / "labels"

OUTPUT_ROOT = (
    BASELINE_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person_failure_visuals"
    / "analyze_person_failure_visuals_v1"
)

IMAGES_OUTPUT = OUTPUT_ROOT / "images"
REPORTS_OUTPUT = OUTPUT_ROOT / "reports"

CSV_OUTPUT = REPORTS_OUTPUT / "person_failure_visuals_v1.csv"
SUMMARY_OUTPUT = REPORTS_OUTPUT / "PERSON_FAILURE_VISUALS_V1_SUMMARY.txt"


# ------------------------------------------------------------
# PARÁMETROS
# ------------------------------------------------------------

PERSON_CLASS = 0
VEHICLE_CLASS = 1

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

TOP_N = 50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------

def load_yolo_labels(label_path, image_width, image_height):
    """
    Lee labels YOLO:
        class cx cy w h

    Devuelve:
        [
            {
                "class_id": int,
                "bbox": [x1, y1, x2, y2],
                "area": float,
            }
        ]
    """

    objects = []

    if not label_path.exists():
        return objects

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return objects

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))

            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            continue

        x1 = (cx - w / 2.0) * image_width
        y1 = (cy - h / 2.0) * image_height
        x2 = (cx + w / 2.0) * image_width
        y2 = (cy + h / 2.0) * image_height

        x1 = max(0.0, min(float(image_width - 1), x1))
        y1 = max(0.0, min(float(image_height - 1), y1))
        x2 = max(0.0, min(float(image_width - 1), x2))
        y2 = max(0.0, min(float(image_height - 1), y2))

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        objects.append(
            {
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2],
                "area": area,
            }
        )

    return objects


def bbox_iou(box_a, box_b):
    """
    IoU entre dos cajas.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def classify_scale(area):
    if area < 16:
        return "<16"
    if area < 32:
        return "16-32"
    if area < 64:
        return "32-64"
    if area < 128:
        return "64-128"
    if area < 256:
        return "128-256"
    if area < 512:
        return "256-512"
    if area < 1024:
        return "512-1024"
    if area < 2048:
        return "1024-2048"
    return ">2048"


def classify_density(objects):
    if objects < 25:
        return "<25"
    if objects < 50:
        return "25-49"
    if objects < 100:
        return "50-99"
    if objects < 200:
        return "100-199"
    if objects < 300:
        return "200-299"
    if objects < 500:
        return "300-499"
    return ">=500"


def draw_box(
    image,
    bbox,
    color,
    label,
    thickness=2,
):
    """
    Dibuja bbox + texto.
    """

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    if label:

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.45
        text_thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            scale,
            text_thickness,
        )

        text_y = max(th + baseline + 2, y1)

        cv2.rectangle(
            image,
            (x1, text_y - th - baseline - 2),
            (x1 + tw + 4, text_y),
            color,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1 + 2, text_y - baseline - 1),
            font,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def create_header(image, text_lines):

    height, width = image.shape[:2]

    header_height = 30 + 25 * len(text_lines)

    canvas = np.zeros(
        (height + header_height, width, 3),
        dtype=np.uint8,
    )

    canvas[header_height:, :] = image

    y = 25

    for line in text_lines:

        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 25

    return canvas


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("# SAR YOLO26 - PERSON FAILURE VISUALS ANALYSIS V1")
    print("=" * 72)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Modelo:")
    print(MODEL_PATH)

    print()
    print("Test:")
    print(TEST_IMAGES)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()

    # --------------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado:\n{DATASET_ROOT}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Directorio test_dev/images no encontrado:\n{TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"Directorio test_dev/labels no encontrado:\n{TEST_LABELS}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # CREAR OUTPUT
    # --------------------------------------------------------

    IMAGES_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # IMÁGENES
    # --------------------------------------------------------

    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print(
        f"Imágenes encontradas: {len(image_paths)}"
    )

    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    print()
    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")

    print()
    print("Analizando PERSON failures...")
    print()

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    image_results = []

    total_gt_person = 0
    total_tp_person = 0
    total_fn_person = 0

    # --------------------------------------------------------
    # PROCESAMIENTO
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        gt_objects = load_yolo_labels(
            label_path,
            image_width,
            image_height,
        )

        gt_persons = [
            obj
            for obj in gt_objects
            if obj["class_id"] == PERSON_CLASS
        ]

        gt_vehicles = [
            obj
            for obj in gt_objects
            if obj["class_id"] == VEHICLE_CLASS
        ]

        total_objects = len(gt_objects)

        # ----------------------------------------------------
        # PREDICCIÓN
        # ----------------------------------------------------

        results = model.predict(
            source=image,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for bbox, cls, conf in zip(
                boxes,
                classes,
                confidences,
            ):

                predictions.append(
                    {
                        "class_id": int(cls),
                        "bbox": bbox.tolist(),
                        "confidence": float(conf),
                    }
                )

        pred_persons = [
            p
            for p in predictions
            if p["class_id"] == PERSON_CLASS
        ]

        pred_vehicles = [
            p
            for p in predictions
            if p["class_id"] == VEHICLE_CLASS
        ]

        # ----------------------------------------------------
        # MATCH PERSON
        # ----------------------------------------------------

        matched_predictions = set()

        tp_persons = []
        fn_persons = []

        for gt in gt_persons:

            best_iou = 0.0
            best_index = None

            for pred_index, pred in enumerate(
                pred_persons
            ):

                if pred_index in matched_predictions:
                    continue

                iou = bbox_iou(
                    gt["bbox"],
                    pred["bbox"],
                )

                if iou > best_iou:

                    best_iou = iou
                    best_index = pred_index

            if (
                best_index is not None
                and best_iou >= IOU_THRESHOLD
            ):

                matched_predictions.add(
                    best_index
                )

                tp_persons.append(
                    {
                        "gt": gt,
                        "prediction": pred_persons[
                            best_index
                        ],
                        "iou": best_iou,
                    }
                )

            else:

                fn_persons.append(
                    {
                        "gt": gt,
                        "iou": best_iou,
                    }
                )

        tp_count = len(tp_persons)
        fn_count = len(fn_persons)

        gt_count = len(gt_persons)

        recall = (
            tp_count / gt_count
            if gt_count > 0
            else 0.0
        )

        total_gt_person += gt_count
        total_tp_person += tp_count
        total_fn_person += fn_count

        # ----------------------------------------------------
        # SOLO NOS INTERESAN IMÁGENES CON FN
        # ----------------------------------------------------

        if fn_count == 0:
            continue

        person_areas = [
            obj["area"]
            for obj in gt_persons
        ]

        median_area = (
            float(np.median(person_areas))
            if person_areas
            else 0.0
        )

        small_16 = sum(
            1
            for area in person_areas
            if area < 16
        )

        small_32 = sum(
            1
            for area in person_areas
            if area < 32
        )

        small_64 = sum(
            1
            for area in person_areas
            if area < 64
        )

        # ----------------------------------------------------
        # SCORE DE FAILURE
        # ----------------------------------------------------

        failure_score = (
            fn_count
            * (1.0 - recall)
            * (
                1.0
                + min(
                    median_area and
                    (64.0 / median_area),
                    10.0,
                )
            )
        )

        image_results.append(
            {
                "image": image_path,
                "gt_person": gt_count,
                "tp_person": tp_count,
                "fn_person": fn_count,
                "recall": recall,
                "objects": total_objects,
                "density": classify_density(
                    total_objects
                ),
                "median_area": median_area,
                "small_16": small_16,
                "small_32": small_32,
                "small_64": small_64,
                "width": image_width,
                "height": image_height,
                "failure_score": failure_score,
                "gt_person_objects": gt_persons,
                "gt_vehicle_objects": gt_vehicles,
                "predictions": predictions,
                "tp_persons": tp_persons,
                "fn_persons": fn_persons,
            }
        )

        if index % 100 == 0 or index == len(image_paths):

            print(
                f"Analizadas: {index:,}/{len(image_paths):,}"
            )

    # --------------------------------------------------------
    # ORDENAR HOTSPOTS
    # --------------------------------------------------------

    image_results.sort(
        key=lambda x: x["failure_score"],
        reverse=True,
    )

    top_results = image_results[:TOP_N]

    print()
    print("=" * 72)
    print("# TOP PERSON FAILURE VISUALS")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # GENERAR VISUALES
    # --------------------------------------------------------

    csv_rows = []

    for rank, data in enumerate(
        top_results,
        start=1,
    ):

        image_path = data["image"]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # ----------------------------------------------------
        # GT PERSON
        # ----------------------------------------------------

        tp_gt_boxes = {
            id(item["gt"])
            for item in data["tp_persons"]
        }

        fn_gt_boxes = {
            id(item["gt"])
            for item in data["fn_persons"]
        }

        for gt in data["gt_person_objects"]:

            if id(gt) in tp_gt_boxes:

                draw_box(
                    image,
                    gt["bbox"],
                    (0, 180, 0),
                    "GT PERSON TP",
                    thickness=2,
                )

            elif id(gt) in fn_gt_boxes:

                draw_box(
                    image,
                    gt["bbox"],
                    (0, 0, 255),
                    "PERSON FN",
                    thickness=2,
                )

        # ----------------------------------------------------
        # VEHICLE GT
        # ----------------------------------------------------

        for vehicle in data["gt_vehicle_objects"]:

            draw_box(
                image,
                vehicle["bbox"],
                (255, 0, 0),
                "GT VEHICLE",
                thickness=1,
            )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        matched_pred_ids = {
            id(item["prediction"])
            for item in data["tp_persons"]
        }

        for pred in data["predictions"]:

            if pred["class_id"] == PERSON_CLASS:

                if id(pred) in matched_pred_ids:

                    color = (0, 255, 255)
                    label = (
                        f"PRED PERSON "
                        f"{pred['confidence']:.2f}"
                    )

                else:

                    color = (0, 140, 255)
                    label = (
                        f"FP PERSON "
                        f"{pred['confidence']:.2f}"
                    )

                draw_box(
                    image,
                    pred["bbox"],
                    color,
                    label,
                    thickness=2,
                )

            elif pred["class_id"] == VEHICLE_CLASS:

                draw_box(
                    image,
                    pred["bbox"],
                    (255, 0, 255),
                    f"PRED VEHICLE "
                    f"{pred['confidence']:.2f}",
                    thickness=1,
                )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = [
            f"Rank: {rank}",
            f"Image: {image_path.name}",
            (
                f"PERSON GT={data['gt_person']} "
                f"TP={data['tp_person']} "
                f"FN={data['fn_person']} "
                f"Recall={data['recall']:.4f}"
            ),
            (
                f"Objects={data['objects']} "
                f"Density={data['density']} "
                f"MedianArea={data['median_area']:.1f}"
            ),
            (
                f"<16={data['small_16']} "
                f"<32={data['small_32']} "
                f"<64={data['small_64']}"
            ),
        ]

        image_with_header = create_header(
            image,
            header,
        )

        output_name = (
            f"top_{rank:03d}_"
            f"{image_path.stem}.jpg"
        )

        output_path = (
            IMAGES_OUTPUT
            / output_name
        )

        cv2.imwrite(
            str(output_path),
            image_with_header,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                95,
            ],
        )

        csv_rows.append(
            {
                "rank": rank,
                "image": image_path.name,
                "visual": output_name,
                "width": data["width"],
                "height": data["height"],
                "gt_person": data["gt_person"],
                "tp_person": data["tp_person"],
                "fn_person": data["fn_person"],
                "recall": data["recall"],
                "objects": data["objects"],
                "density": data["density"],
                "median_area": data["median_area"],
                "small_16": data["small_16"],
                "small_32": data["small_32"],
                "small_64": data["small_64"],
                "failure_score": data["failure_score"],
            }
        )

        print(
            f"{rank:02d}. "
            f"{image_path.name} "
            f"GT={data['gt_person']:4d} "
            f"TP={data['tp_person']:4d} "
            f"FN={data['fn_person']:4d} "
            f"Recall={data['recall']:.4f}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fieldnames = [
        "rank",
        "image",
        "visual",
        "width",
        "height",
        "gt_person",
        "tp_person",
        "fn_person",
        "recall",
        "objects",
        "density",
        "median_area",
        "small_16",
        "small_32",
        "small_64",
        "failure_score",
    ]

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(csv_rows)

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    global_recall = (
        total_tp_person / total_gt_person
        if total_gt_person > 0
        else 0.0
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON FAILURE VISUALS "
            "ANALYSIS V1\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            f"Dataset:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Modelo:\n{MODEL_PATH}\n\n"
        )

        f.write(
            f"Test:\n{TEST_IMAGES}\n\n"
        )

        f.write(
            f"Imágenes test: {len(image_paths):,}\n"
        )

        f.write(
            f"PERSON GT: {total_gt_person:,}\n"
        )

        f.write(
            f"PERSON TP: {total_tp_person:,}\n"
        )

        f.write(
            f"PERSON FN: {total_fn_person:,}\n"
        )

        f.write(
            f"PERSON Recall: {global_recall:.4f}\n\n"
        )

        f.write(
            f"Hotspots visualizados: "
            f"{len(csv_rows)}\n\n"
        )

        f.write(
            "INTERPRETACIÓN\n"
        )

        f.write(
            "Las imágenes generadas corresponden a los "
            "principales hotspots de fallo de PERSON.\n"
        )

        f.write(
            "Se muestran simultáneamente Ground Truth, "
            "true positives, false negatives y "
            "predicciones del modelo.\n\n"
        )

        f.write(
            "Este análisis es exclusivamente diagnóstico. "
            "No modifica el dataset original ni el modelo.\n"
        )

    print()
    print("=" * 72)
    print("# RESULTADO PERSON FAILURE VISUALS V1")
    print("=" * 72)
    print()

    print(
        f"Imágenes test:       {len(image_paths):,}"
    )

    print(
        f"PERSON GT:           {total_gt_person:,}"
    )

    print(
        f"PERSON TP:           {total_tp_person:,}"
    )

    print(
        f"PERSON FN:           {total_fn_person:,}"
    )

    print(
        f"PERSON Recall:       {global_recall:.4f}"
    )

    print()

    print(
        f"Hotspots visualizados: {len(csv_rows)}"
    )

    print()

    print("[OK]", CSV_OUTPUT)
    print("[OK]", SUMMARY_OUTPUT)
    print("[OK]", IMAGES_OUTPUT)

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()