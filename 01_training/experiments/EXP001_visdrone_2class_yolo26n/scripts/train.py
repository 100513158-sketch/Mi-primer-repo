from pathlib import Path
import sys
import shutil
import yaml

from ultralytics import YOLO


# ============================================================
# IDENTIFICACIÓN DEL EXPERIMENTO
# ============================================================

EXPERIMENT_NAME = "EXP001_visdrone_2class_yolo26n"


# ============================================================
# RUTAS
# ============================================================

# train.py
SCRIPT_DIR = Path(__file__).resolve().parent

# EXP001_visdrone_2class_yolo26n
EXPERIMENT_DIR = SCRIPT_DIR.parent

# 01_training
TRAINING_DIR = EXPERIMENT_DIR.parent.parent

# Proyecto SARC-Drone
PROJECT_DIR = TRAINING_DIR.parent


CONFIG_DIR = EXPERIMENT_DIR / "config"
RUNS_DIR = EXPERIMENT_DIR / "runs"
LOGS_DIR = EXPERIMENT_DIR / "logs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_YAML = (
    PROJECT_DIR
    / "00_DATASETS"
    / "SAR_DATASET_STUDIO"
    / "configs"
    / "sar_visdrone_2class.yaml"
)

MODEL_SOURCE = (
    TRAINING_DIR
    / "models"
    / "pretrained"
    / "yolo26n.pt"
)


# ============================================================
# PARÁMETROS DEL ENTRENAMIENTO
# ============================================================

IMG_SIZE = 640
EPOCHS = 3
BATCH = 8
DEVICE = 0
WORKERS = 4

RUN_NAME = "baseline_3epochs"


# ============================================================
# FUNCIONES
# ============================================================

def print_header():
    print("=" * 70)
    print("SARC-DRONE - ENTRENAMIENTO YOLO")
    print("=" * 70)
    print(f"Experimento : {EXPERIMENT_NAME}")
    print(f"Dataset     : {DATASET_YAML}")
    print(f"Modelo      : {MODEL_SOURCE}")
    print(f"Imagen      : {IMG_SIZE}")
    print(f"Épocas      : {EPOCHS}")
    print(f"Batch       : {BATCH}")
    print(f"GPU         : {DEVICE}")
    print(f"Workers     : {WORKERS}")
    print("=" * 70)
    print()


def check_paths():
    print("[1] VERIFICACIÓN DE RUTAS")

    if not DATASET_YAML.exists():
        print(f"ERROR: no existe el YAML:")
        print(DATASET_YAML)
        sys.exit(1)

    print("OK: dataset YAML encontrado")

    if not MODEL_SOURCE.exists():
        print(f"ERROR: no existe el modelo:")
        print(MODEL_SOURCE)
        sys.exit(1)

    print("OK: modelo YOLO encontrado")

    print()


def create_directories():
    print("[2] CREACIÓN DE DIRECTORIOS")

    for directory in [
        RUNS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        CHECKPOINTS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print("OK: estructura de salida preparada")
    print()


def validate_yaml():
    print("[3] VALIDACIÓN DEL DATASET")

    try:
        with open(
            DATASET_YAML,
            "r",
            encoding="utf-8"
        ) as f:
            data = yaml.safe_load(f)

    except Exception as exc:
        print("ERROR leyendo YAML:")
        print(exc)
        sys.exit(1)

    required = [
        "path",
        "train",
        "val",
        "names",
    ]

    for key in required:
        if key not in data:
            print(f"ERROR: falta '{key}' en el YAML")
            sys.exit(1)

    dataset_path = Path(data["path"])

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_DIR / dataset_path

    if not dataset_path.exists():
        print("ERROR: no existe el dataset:")
        print(dataset_path)
        sys.exit(1)

    print(f"OK: dataset encontrado")
    print(f"Path: {dataset_path}")

    print(f"Train: {data['train']}")
    print(f"Val  : {data['val']}")

    if "test" in data:
        print(f"Test : {data['test']}")

    print(f"Clases: {data['names']}")

    print()


def save_experiment_config():
    print("[4] GUARDANDO CONFIGURACIÓN")

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config_file = CONFIG_DIR / "experiment.yaml"

    config = {
        "experiment": EXPERIMENT_NAME,
        "dataset_yaml": str(DATASET_YAML),
        "model": str(MODEL_SOURCE),
        "image_size": IMG_SIZE,
        "epochs": EPOCHS,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "run_name": RUN_NAME,
    }

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"OK: configuración guardada:")
    print(config_file)
    print()


def train():
    print("[5] CARGANDO MODELO")

    model = YOLO(str(MODEL_SOURCE))

    print("OK: modelo cargado")
    print()

    print("[6] INICIANDO ENTRENAMIENTO")
    print()

    results = model.train(
        data=str(DATASET_YAML),

        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,

        device=DEVICE,
        workers=WORKERS,

        project=str(RUNS_DIR),
        name=RUN_NAME,

        pretrained=True,

        exist_ok=False,

        verbose=True,

        plots=True,

        save=True,
        save_period=-1,

        patience=50,

        amp=True,

        cache=False,

        deterministic=True,
    )

    return results


def copy_checkpoints():
    print()
    print("[7] ORGANIZANDO CHECKPOINTS")

    run_dir = RUNS_DIR / RUN_NAME
    weights_dir = run_dir / "weights"

    if not weights_dir.exists():
        print("ADVERTENCIA: no se encontró weights/")
        return

    for filename in [
        "best.pt",
        "last.pt",
    ]:

        source = weights_dir / filename

        if source.exists():

            destination = CHECKPOINTS_DIR / filename

            shutil.copy2(
                source,
                destination
            )

            print(
                f"OK: {filename} -> "
                f"{destination}"
            )

    print()


def print_summary():
    run_dir = RUNS_DIR / RUN_NAME

    print("=" * 70)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 70)

    print()
    print(f"Experimento:")
    print(EXPERIMENT_NAME)

    print()
    print("Resultados:")
    print(run_dir)

    print()
    print("Checkpoints:")
    print(CHECKPOINTS_DIR)

    print()
    print("Archivos importantes:")

    for file in [
        run_dir / "results.csv",
        run_dir / "results.png",
        run_dir / "confusion_matrix.png",
        CHECKPOINTS_DIR / "best.pt",
        CHECKPOINTS_DIR / "last.pt",
    ]:

        if file.exists():
            print(f"OK  {file}")
        else:
            print(f"--  {file}")

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    check_paths()

    create_directories()

    validate_yaml()

    save_experiment_config()

    train()

    copy_checkpoints()

    print_summary()


if __name__ == "__main__":
    main()