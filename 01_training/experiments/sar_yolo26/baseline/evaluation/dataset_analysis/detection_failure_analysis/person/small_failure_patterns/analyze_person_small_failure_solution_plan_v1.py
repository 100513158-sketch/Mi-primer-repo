from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# SAR YOLO26
# PERSON SMALL FAILURE SOLUTION PLAN ANALYSIS V1
# ============================================================================


SCRIPT_PATH = Path(__file__).resolve()

# .../baseline/evaluation/dataset_analysis/...
BASE_DIR = SCRIPT_PATH.parents[5]

REPORTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)


# ============================================================================
# INPUT REPORTS
# ============================================================================

INPUT_FILES = {
    "solution_targets":
        "person_small_failure_solution_targets_v1.csv",

    "solution_ranking":
        "person_small_failure_solution_targets_ranking_v1.csv",

    "solution_interactions":
        "person_small_failure_solution_targets_interactions_v1.csv",

    "solution_pairs":
        "person_small_failure_solution_targets_factor_pairs_v1.csv",

    "factor_contribution":
        "person_small_failure_factor_contribution_v1.csv",

    "factor_priority":
        "person_small_failure_priority_v1.csv",

    "factor_isolation":
        "person_small_failure_factor_isolation_v1.csv",

    "factor_consistency":
        "person_small_failure_factor_consistency_v1.csv",
}


# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_FILES = {
    "factor_plan":
        "person_small_failure_solution_plan_v1.csv",

    "factor_plan_ranking":
        "person_small_failure_solution_plan_ranking_v1.csv",

    "interaction_plan":
        "person_small_failure_solution_plan_interactions_v1.csv",

    "experiment_plan":
        "person_small_failure_solution_experiment_plan_v1.csv",

    "summary":
        "PERSON_SMALL_FAILURE_SOLUTION_PLAN_V1_SUMMARY.txt",
}


# ============================================================================
# FACTOR DEFINITIONS
# ============================================================================

FACTOR_ACTIONS = {

    "NO_PREDICTION": {
        "severity": "CRITICAL",

        "target":
            "INCREASE_DETECTION_RECALL",

        "problem":
            "La persona pequeña no genera una predicción válida.",

        "interventions": [
            "Aumentar la representación de personas pequeñas en entrenamiento.",
            "Aplicar oversampling dirigido a objetos pequeños.",
            "Introducir crops/tiles que preserven personas pequeñas.",
            "Revisar la resolución efectiva de entrada.",
            "Evaluar entrenamiento con mayor resolución.",
            "Analizar recall específico de PERSON-small."
        ],

        "metrics": [
            "PERSON small recall",
            "PERSON small FN rate",
            "PERSON small no-prediction rate"
        ]
    },

    "DENSE_SCENE": {
        "severity": "CRITICAL",

        "target":
            "IMPROVE_DENSE_SCENE_DETECTION",

        "problem":
            "Las personas pequeñas aparecen en escenas con alta densidad de objetos.",

        "interventions": [
            "Aumentar muestras de escenas densas.",
            "Oversampling de imágenes con múltiples personas.",
            "Introducir crops locales de regiones densas.",
            "Evaluar tiling/inference por regiones.",
            "Revisar separación entre personas próximas.",
            "Evaluar recall específicamente en escenas densas."
        ],

        "metrics": [
            "Dense-scene recall",
            "Dense-scene FN rate",
            "Small-person recall in dense scenes"
        ]
    },

    "EDGE_LOCATION": {
        "severity": "HIGH",

        "target":
            "IMPROVE_BORDER_AWARE_DETECTION",

        "problem":
            "Las personas pequeñas situadas cerca de los bordes presentan más fallos.",

        "interventions": [
            "Aumentar muestras de personas pequeñas cerca de bordes.",
            "Aplicar crops con padding.",
            "Evitar pérdida de contexto durante cropping.",
            "Introducir augmentations de traslación controlada.",
            "Evaluar inferencia con solapamiento entre tiles."
        ],

        "metrics": [
            "Border recall",
            "Border FN rate",
            "Small-person border recall"
        ]
    },

    "CLOSE_NEIGHBORS": {
        "severity": "HIGH",

        "target":
            "IMPROVE_PERSON_SEPARATION",

        "problem":
            "Personas pequeñas próximas entre sí dificultan la separación de instancias.",

        "interventions": [
            "Aumentar escenas con personas próximas.",
            "Oversampling de grupos densos de personas.",
            "Revisar resolución de entrada.",
            "Evaluar tiling.",
            "Revisar configuración de NMS.",
            "Analizar errores de separación entre personas."
        ],

        "metrics": [
            "Close-neighbor recall",
            "Close-neighbor FN rate",
            "Person separation errors"
        ]
    },

    "EXTREME_SMALL": {
        "severity": "HIGH",

        "target":
            "IMPROVE_EXTREME_SMALL_OBJECT_DETECTION",

        "problem":
            "Los objetos extremadamente pequeños son difíciles de representar espacialmente.",

        "interventions": [
            "Oversampling de objetos extremadamente pequeños.",
            "Aumentar resolución de entrenamiento.",
            "Usar crops/tiles centrados en objetos pequeños.",
            "Evitar augmentations que reduzcan todavía más el objeto.",
            "Evaluar escalado mínimo de objetos pequeños."
        ],

        "metrics": [
            "Extreme-small recall",
            "Extreme-small FN rate",
            "Recall by object-size bucket"
        ]
    },

    "LOCALIZATION_ERROR": {
        "severity": "MEDIUM",

        "target":
            "IMPROVE_LOCALIZATION",

        "problem":
            "Existe predicción, pero la localización de la persona no es suficientemente precisa.",

        "interventions": [
            "Analizar distribución de IoU.",
            "Revisar calidad de bounding boxes.",
            "Aumentar ejemplos de objetos pequeños con localización difícil.",
            "Evaluar pérdidas de bounding box.",
            "Comparar localization error por tamaño."
        ],

        "metrics": [
            "Small-person IoU",
            "Localization error rate",
            "IoU distribution"
        ]
    },

    "OCCLUSION": {
        "severity": "MEDIUM",

        "target":
            "IMPROVE_OCCLUSION_ROBUSTNESS",

        "problem":
            "Las personas parcialmente ocultas presentan mayor probabilidad de fallo.",

        "interventions": [
            "Aumentar muestras de personas parcialmente ocluidas.",
            "Aplicar augmentations de oclusión controlada.",
            "Mantener ejemplos reales de oclusión.",
            "Evaluar recall por nivel de oclusión.",
            "Evitar augmentations que destruyan completamente el objeto."
        ],

        "metrics": [
            "Occluded-person recall",
            "Occlusion FN rate",
            "Recall by occlusion level"
        ]
    },
}


# ============================================================================
# UTILITIES
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"# {title}")
    print("=" * 72)
    print()


def normalize(value: str) -> str:
    return str(value).strip()


def safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default

        text = str(value).strip()

        if not text:
            return default

        return float(text.replace(",", "."))

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV sin cabecera:\n{path}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV vacío:\n{path}"
        )

    return rows


def write_csv(
    path: Path,
    rows: List[Dict]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def percentage(
    value: float,
    total: float
) -> float:

    if total == 0:
        return 0.0

    return value / total * 100.0


# ============================================================================
# REPORT CHECK
# ============================================================================

def check_reports() -> None:

    print("Comprobando reports de entrada...")

    missing = []

    for key, filename in INPUT_FILES.items():

        path = REPORTS_DIR / filename

        if path.exists() and path.stat().st_size > 0:

            print(f"[OK] {key}")

        else:

            print(f"[MISSING] {key}")
            missing.append(
                (key, path)
            )

    if missing:

        print()
        print("REPORTS FALTANTES:")
        print()

        for key, path in missing:

            print(key)
            print(path)
            print()

        raise FileNotFoundError(
            "Faltan uno o más reports necesarios "
            "para generar el solution plan."
        )

    print()
    print("[OK] Todos los reports de entrada están disponibles.")


# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_reports():

    print()
    print("Cargando solution targets...")

    solution_targets = load_csv(
        REPORTS_DIR / INPUT_FILES["solution_targets"]
    )

    print(
        f"[OK] Solution targets: "
        f"{len(solution_targets)}"
    )

    print()
    print("Cargando solution ranking...")

    solution_ranking = load_csv(
        REPORTS_DIR / INPUT_FILES["solution_ranking"]
    )

    print(
        f"[OK] Solution ranking: "
        f"{len(solution_ranking)}"
    )

    print()
    print("Cargando interacciones...")

    solution_interactions = load_csv(
        REPORTS_DIR / INPUT_FILES["solution_interactions"]
    )

    print(
        f"[OK] Interacciones: "
        f"{len(solution_interactions)}"
    )

    print()
    print("Cargando contribución...")

    contribution = load_csv(
        REPORTS_DIR / INPUT_FILES["factor_contribution"]
    )

    print(
        f"[OK] Contribution: "
        f"{len(contribution)}"
    )

    print()
    print("Cargando prioridad...")

    priority = load_csv(
        REPORTS_DIR / INPUT_FILES["factor_priority"]
    )

    print(
        f"[OK] Priority: "
        f"{len(priority)}"
    )

    print()
    print("Cargando aislamiento...")

    isolation = load_csv(
        REPORTS_DIR / INPUT_FILES["factor_isolation"]
    )

    print(
        f"[OK] Isolation: "
        f"{len(isolation)}"
    )

    print()
    print("Cargando consistencia...")

    consistency = load_csv(
        REPORTS_DIR / INPUT_FILES["factor_consistency"]
    )

    print(
        f"[OK] Consistency: "
        f"{len(consistency)}"
    )

    return (
        solution_targets,
        solution_ranking,
        solution_interactions,
        contribution,
        priority,
        isolation,
        consistency
    )


# ============================================================================
# INDEX HELPERS
# ============================================================================

def index_by_factor(
    rows: List[Dict]
) -> Dict[str, Dict]:

    result = {}

    for row in rows:

        factor = normalize(
            row.get("factor", "")
        )

        if factor:
            result[factor] = row

    return result


def find_value(
    row: Dict,
    candidates: List[str],
    default=0.0
):

    for key in candidates:

        if key in row:

            return row[key]

    return default


# ============================================================================
# FACTOR SCORE
# ============================================================================

def calculate_plan_score(
    contribution: float,
    priority: float,
    isolation: float,
    consistency: float
) -> float:

    # ------------------------------------------------------------
    # El score combina:
    #
    # 35% contribución
    # 30% prioridad
    # 20% aislamiento
    # 15% consistencia
    #
    # Normalización robusta a 0..100.
    # ------------------------------------------------------------

    contribution = max(
        0.0,
        min(100.0, contribution)
    )

    priority = max(
        0.0,
        min(100.0, priority)
    )

    isolation = max(
        0.0,
        min(100.0, isolation)
    )

    consistency = max(
        0.0,
        min(100.0, consistency)
    )

    score = (
        contribution * 0.35
        + priority * 0.30
        + isolation * 0.20
        + consistency * 0.15
    )

    return score


# ============================================================================
# BUILD FACTOR PLAN
# ============================================================================

def build_factor_plan(
    solution_targets,
    contribution,
    priority,
    isolation,
    consistency
) -> List[Dict]:

    target_index = index_by_factor(
        solution_targets
    )

    contribution_index = index_by_factor(
        contribution
    )

    priority_index = index_by_factor(
        priority
    )

    isolation_index = index_by_factor(
        isolation
    )

    consistency_index = index_by_factor(
        consistency
    )

    common_factors = sorted(
        set(target_index)
        & set(contribution_index)
        & set(priority_index)
        & set(isolation_index)
        & set(consistency_index)
    )

    print()
    print(
        "Construyendo plan de intervención por factor..."
    )

    rows = []

    for factor in common_factors:

        target_row = target_index[factor]
        contribution_row = contribution_index[factor]
        priority_row = priority_index[factor]
        isolation_row = isolation_index[factor]
        consistency_row = consistency_index[factor]

        failures = safe_int(
            find_value(
                target_row,
                ["failures"]
            )
        )

        contribution_pct = safe_float(
            find_value(
                contribution_row,
                [
                    "contribution_percentage",
                    "contribution"
                ]
            )
        )

        priority_score = safe_float(
            find_value(
                priority_row,
                [
                    "priority",
                    "priority_score",
                    "score"
                ]
            )
        )

        isolation_score = safe_float(
            find_value(
                isolation_row,
                [
                    "priority",
                    "isolation",
                    "isolation_score"
                ]
            )
        )

        consistency_score = safe_float(
            find_value(
                consistency_row,
                [
                    "consistency",
                    "consistency_score"
                ]
            )
        )

        solution_score = calculate_plan_score(
            contribution_pct,
            priority_score,
            isolation_score,
            consistency_score
        )

        metadata = FACTOR_ACTIONS.get(
            factor,
            {
                "severity": "UNKNOWN",
                "target": "REVIEW_FACTOR",
                "problem": "Factor no documentado.",
                "interventions": [],
                "metrics": []
            }
        )

        rows.append({

            "factor":
                factor,

            "severity":
                metadata["severity"],

            "failures":
                failures,

            "contribution_percentage":
                round(
                    contribution_pct,
                    6
                ),

            "priority_score":
                round(
                    priority_score,
                    6
                ),

            "isolation_score":
                round(
                    isolation_score,
                    6
                ),

            "consistency_score":
                round(
                    consistency_score,
                    6
                ),

            "solution_score":
                round(
                    solution_score,
                    6
                ),

            "target":
                target_row.get(
                    "target",
                    metadata["target"]
                ),

            "problem":
                metadata["problem"],

            "interventions":
                " | ".join(
                    metadata["interventions"]
                ),

            "metrics":
                " | ".join(
                    metadata["metrics"]
                )
        })

    rows.sort(
        key=lambda row:
            safe_float(
                row["solution_score"]
            ),
        reverse=True
    )

    for index, row in enumerate(
        rows,
        start=1
    ):

        row["plan_rank"] = index

    print(
        f"[OK] Factores en plan: {len(rows)}"
    )

    return rows


# ============================================================================
# INTERACTION ACTIONS
# ============================================================================

INTERACTION_ACTIONS = {

    frozenset([
        "DENSE_SCENE",
        "NO_PREDICTION"
    ]): {
        "severity": "CRITICAL",
        "target": "RECOVER_RECALL_IN_DENSE_SMALL_SCENES",
        "action":
            "Priorizar oversampling de escenas densas + crops/tiles + aumento de resolución.",
        "metrics":
            "Dense small-person recall | FN rate | no-prediction rate"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "DENSE_SCENE"
    ]): {
        "severity": "CRITICAL",
        "target": "IMPROVE_SEPARATION_IN_DENSE_PERSON_GROUPS",
        "action":
            "Entrenar con grupos densos de personas y evaluar separación mediante crops/tiles.",
        "metrics":
            "Close-neighbor recall | separation errors | dense-scene recall"
    },

    frozenset([
        "EDGE_LOCATION",
        "NO_PREDICTION"
    ]): {
        "severity": "HIGH",
        "target": "RECOVER_BORDER_SMALL_OBJECT_RECALL",
        "action":
            "Usar crops con padding y aumentar representación de personas pequeñas en bordes.",
        "metrics":
            "Border recall | border FN rate"
    },

    frozenset([
        "EXTREME_SMALL",
        "NO_PREDICTION"
    ]): {
        "severity": "CRITICAL",
        "target": "RECOVER_EXTREME_SMALL_RECALL",
        "action":
            "Aumentar resolución efectiva y utilizar crops centrados en objetos extremadamente pequeños.",
        "metrics":
            "Extreme-small recall | FN rate by size"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "NO_PREDICTION"
    ]): {
        "severity": "HIGH",
        "target": "RECOVER_NEIGHBOR_SMALL_PERSON_RECALL",
        "action":
            "Aumentar escenas con personas próximas y revisar separación/NMS.",
        "metrics":
            "Close-neighbor recall | FN rate"
    },

    frozenset([
        "DENSE_SCENE",
        "EDGE_LOCATION"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_DENSE_BORDER_RECALL",
        "action":
            "Combinar muestras densas cercanas a bordes con crops con padding.",
        "metrics":
            "Dense-border recall"
    },

    frozenset([
        "DENSE_SCENE",
        "LOCALIZATION_ERROR"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_LOCALIZATION_IN_DENSE_SCENES",
        "action":
            "Reforzar ejemplos con bounding boxes pequeños y próximos en escenas densas.",
        "metrics":
            "IoU | localization error rate"
    },

    frozenset([
        "DENSE_SCENE",
        "EXTREME_SMALL"
    ]): {
        "severity": "CRITICAL",
        "target": "IMPROVE_EXTREME_SMALL_DENSE_RECALL",
        "action":
            "Priorizar crops de alta densidad manteniendo resolución suficiente.",
        "metrics":
            "Extreme-small dense recall"
    },

    frozenset([
        "DENSE_SCENE",
        "OCCLUSION"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_OCCLUDED_DENSE_PERSON_RECALL",
        "action":
            "Aumentar escenas densas con oclusión real y augmentations de oclusión controlada.",
        "metrics":
            "Occluded dense recall"
    },

    frozenset([
        "NO_PREDICTION",
        "OCCLUSION"
    ]): {
        "severity": "HIGH",
        "target": "RECOVER_OCCLUDED_SMALL_PERSON_RECALL",
        "action":
            "Aumentar ejemplos de personas pequeñas ocluidas y preservar contexto.",
        "metrics":
            "Occluded small-person recall"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "EXTREME_SMALL"
    ]): {
        "severity": "HIGH",
        "target": "SEPARATE_EXTREME_SMALL_PERSONS",
        "action":
            "Aumentar resolución y crops de grupos de personas pequeñas.",
        "metrics":
            "Extreme-small neighbor recall"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "EDGE_LOCATION"
    ]): {
        "severity": "HIGH",
        "target": "SEPARATE_BORDER_NEIGHBORS",
        "action":
            "Entrenar con personas próximas en bordes y crops con padding.",
        "metrics":
            "Border-neighbor recall"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "LOCALIZATION_ERROR"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_NEIGHBOR_LOCALIZATION",
        "action":
            "Reforzar localización en grupos de personas próximas.",
        "metrics":
            "IoU | neighbor localization error"
    },

    frozenset([
        "CLOSE_NEIGHBORS",
        "OCCLUSION"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_OCCLUDED_NEIGHBOR_SEPARATION",
        "action":
            "Entrenar con grupos de personas parcialmente ocluidas.",
        "metrics":
            "Occluded-neighbor recall"
    },

    frozenset([
        "LOCALIZATION_ERROR",
        "OCCLUSION"
    ]): {
        "severity": "MEDIUM",
        "target": "IMPROVE_OCCLUDED_LOCALIZATION",
        "action":
            "Reforzar localización en personas parcialmente visibles.",
        "metrics":
            "Occluded IoU | localization error"
    },

    frozenset([
        "EDGE_LOCATION",
        "EXTREME_SMALL"
    ]): {
        "severity": "HIGH",
        "target": "IMPROVE_BORDER_EXTREME_SMALL_RECALL",
        "action":
            "Aumentar muestras extremadamente pequeñas cerca de bordes.",
        "metrics":
            "Extreme-small border recall"
    },
}


# ============================================================================
# BUILD INTERACTION PLAN
# ============================================================================

def build_interaction_plan(
    solution_interactions
) -> List[Dict]:

    print()
    print(
        "Construyendo plan de intervención "
        "por interacción..."
    )

    rows = []

    for source_row in solution_interactions:

        factor_a = normalize(
            source_row.get(
                "factor_a",
                ""
            )
        )

        factor_b = normalize(
            source_row.get(
                "factor_b",
                ""
            )
        )

        if not factor_a or not factor_b:
            continue

        key = frozenset([
            factor_a,
            factor_b
        ])

        metadata = INTERACTION_ACTIONS.get(
            key,
            {
                "severity": "MEDIUM",
                "target": "REVIEW_INTERACTION",
                "action":
                    "Analizar conjuntamente ambos factores.",
                "metrics":
                    "Joint recall"
            }
        )

        failures = safe_int(
            source_row.get(
                "failures",
                0
            )
        )

        percentage_total = safe_float(
            source_row.get(
                "percentage_of_total_failures",
                source_row.get(
                    "percentage",
                    0
                )
            )
        )

        score = safe_float(
            source_row.get(
                "score",
                source_row.get(
                    "priority",
                    0
                )
            )
        )

        rows.append({

            "factor_a":
                factor_a,

            "factor_b":
                factor_b,

            "interaction":
                source_row.get(
                    "interaction",
                    f"{factor_a} + {factor_b}"
                ),

            "severity":
                metadata["severity"],

            "failures":
                failures,

            "percentage_of_total_failures":
                round(
                    percentage_total,
                    6
                ),

            "solution_score":
                round(
                    score,
                    6
                ),

            "target":
                metadata["target"],

            "action":
                metadata["action"],

            "metrics":
                metadata["metrics"]
        })

    rows.sort(
        key=lambda row:
            (
                safe_float(
                    row["solution_score"]
                ),
                safe_int(
                    row["failures"]
                )
            ),
        reverse=True
    )

    for index, row in enumerate(
        rows,
        start=1
    ):

        row["plan_rank"] = index

    print(
        f"[OK] Interacciones en plan: {len(rows)}"
    )

    return rows


# ============================================================================
# EXPERIMENT PLAN
# ============================================================================

def build_experiment_plan(
    factor_plan,
    interaction_plan
) -> List[Dict]:

    print()
    print(
        "Construyendo plan experimental..."
    )

    experiments = []

    # ------------------------------------------------------------
    # EXPERIMENTO 1
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP01",

        "priority":
            1,

        "name":
            "SMALL_PERSON_RECALL_BASELINE",

        "objective":
            "Medir baseline específico de PERSON pequeños.",

        "changes":
            "NINGUNA",

        "dataset_change":
            "NO",

        "training_change":
            "NO",

        "metrics":
            "PERSON small recall | FN rate | no-prediction rate",

        "reason":
            "Necesitamos una referencia específica antes de intervenir."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 2
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP02",

        "priority":
            2,

        "name":
            "TARGETED_SMALL_PERSON_OVERSAMPLING",

        "objective":
            "Aumentar exposición del modelo a personas pequeñas.",

        "changes":
            "Oversampling dirigido a PERSON small/extreme-small.",

        "dataset_change":
            "SI - SOLO INDICE/SAMPLING",

        "training_change":
            "SI",

        "metrics":
            "Small recall | extreme-small recall | overall recall",

        "reason":
            "NO_PREDICTION y EXTREME_SMALL son problemas prioritarios."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 3
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP03",

        "priority":
            3,

        "name":
            "HIGH_RESOLUTION_SMALL_OBJECT",

        "objective":
            "Evaluar si la resolución efectiva limita la detección.",

        "changes":
            "Incrementar imgsz manteniendo el resto constante.",

        "dataset_change":
            "NO",

        "training_change":
            "SI",

        "metrics":
            "Small recall | extreme-small recall | GPU memory | inference time",

        "reason":
            "EXTREME_SMALL + NO_PREDICTION es una interacción crítica."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 4
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP04",

        "priority":
            4,

        "name":
            "DENSE_SCENE_TARGETED_CROPS",

        "objective":
            "Mejorar detección en escenas densas.",

        "changes":
            "Crops/tiles dirigidos a escenas densas.",

        "dataset_change":
            "SI - DERIVADO, NO DESTRUIR ORIGINAL",

        "training_change":
            "SI",

        "metrics":
            "Dense recall | dense small recall | neighbor recall",

        "reason":
            "DENSE_SCENE + NO_PREDICTION es la interacción número 1."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 5
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP05",

        "priority":
            5,

        "name":
            "BORDER_PADDING",

        "objective":
            "Reducir fallos de objetos pequeños próximos a bordes.",

        "changes":
            "Crops con padding y traslación controlada.",

        "dataset_change":
            "SI - DERIVADO",

        "training_change":
            "SI",

        "metrics":
            "Border recall | border FN rate | small border recall",

        "reason":
            "EDGE_LOCATION + NO_PREDICTION tiene 31.67% de fallos."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 6
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP06",

        "priority":
            6,

        "name":
            "NEIGHBOR_SEPARATION",

        "objective":
            "Mejorar separación de personas próximas.",

        "changes":
            "Entrenamiento dirigido a grupos densos y revisión de NMS.",

        "dataset_change":
            "SI - DERIVADO",

        "training_change":
            "SI",

        "metrics":
            "Close-neighbor recall | separation errors",

        "reason":
            "CLOSE_NEIGHBORS aparece entre los principales factores."
    })

    # ------------------------------------------------------------
    # EXPERIMENTO 7
    # ------------------------------------------------------------

    experiments.append({

        "experiment_id":
            "EXP07",

        "priority":
            7,

        "name":
            "COMBINED_TARGETED_PIPELINE",

        "objective":
            "Combinar las intervenciones que hayan demostrado mejora.",

        "changes":
            "Oversampling + resolución + crops densos + border padding.",

        "dataset_change":
            "SI - DERIVADO",

        "training_change":
            "SI",

        "metrics":
            "Small recall | dense recall | border recall | neighbor recall",

        "reason":
            "Solo después de validar individualmente cada intervención."
    })

    return experiments


# ============================================================================
# SUMMARY
# ============================================================================

def generate_summary(
    factor_plan,
    interaction_plan,
    experiment_plan
) -> str:

    lines = []

    lines.append(
        "PERSON SMALL FAILURE SOLUTION PLAN V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "OBJETIVO"
    )

    lines.append(
        "Convertir el análisis de fallos en un plan experimental "
        "dirigido para mejorar PERSON small detection."
    )

    lines.append("")

    lines.append(
        "CONCLUSION PRINCIPAL"
    )

    lines.append(
        "El problema dominante es la ausencia de predicción "
        "en personas pequeñas, especialmente cuando coinciden "
        "escenas densas, proximidad entre personas y objetos "
        "extremadamente pequeños."
    )

    lines.append("")

    lines.append(
        "TOP FACTORES"
    )

    lines.append(
        "-" * 72
    )

    for row in factor_plan:

        lines.append(
            f"{row['plan_rank']:2d}. "
            f"{row['factor']:<24} "
            f"Failures={row['failures']:6d} "
            f"Contribution={row['contribution_percentage']:7.2f}% "
            f"Score={row['solution_score']:7.2f} "
            f"Severity={row['severity']}"
        )

        lines.append(
            f"    TARGET: {row['target']}"
        )

    lines.append("")

    lines.append(
        "TOP INTERACCIONES"
    )

    lines.append(
        "-" * 72
    )

    for row in interaction_plan[:15]:

        lines.append(
            f"{row['plan_rank']:2d}. "
            f"{row['factor_a']} + "
            f"{row['factor_b']} "
            f"Failures={row['failures']:6d} "
            f"Percentage={row['percentage_of_total_failures']:7.2f}% "
            f"Severity={row['severity']}"
        )

        lines.append(
            f"    TARGET: {row['target']}"
        )

    lines.append("")

    lines.append(
        "PLAN EXPERIMENTAL"
    )

    lines.append(
        "-" * 72
    )

    for experiment in experiment_plan:

        lines.append(
            f"{experiment['experiment_id']} | "
            f"{experiment['name']}"
        )

        lines.append(
            f"  OBJETIVO: "
            f"{experiment['objective']}"
        )

        lines.append(
            f"  CAMBIOS: "
            f"{experiment['changes']}"
        )

        lines.append(
            f"  METRICAS: "
            f"{experiment['metrics']}"
        )

        lines.append(
            f"  MOTIVO: "
            f"{experiment['reason']}"
        )

        lines.append("")

    lines.append(
        "REGLA EXPERIMENTAL"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "No combinar varias intervenciones inicialmente. "
        "Cada cambio debe compararse contra el baseline "
        "para determinar su efecto real."
    )

    lines.append("")

    lines.append(
        "ORDEN RECOMENDADO"
    )

    lines.append(
        "1. Baseline específico PERSON-small"
    )

    lines.append(
        "2. Oversampling dirigido"
    )

    lines.append(
        "3. Mayor resolución"
    )

    lines.append(
        "4. Crops/tiles para escenas densas"
    )

    lines.append(
        "5. Border padding"
    )

    lines.append(
        "6. Separación de vecinos"
    )

    lines.append(
        "7. Pipeline combinado"
    )

    lines.append("")

    lines.append(
        "IMPORTANTE: este análisis NO modifica el dataset "
        "ni ningún modelo."
    )

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "SAR YOLO26 - PERSON SMALL FAILURE "
        "SOLUTION PLAN ANALYSIS V1"
    )

    print("SCRIPT:")
    print(SCRIPT_PATH)

    print()
    print("BASE:")
    print(BASE_DIR)

    print()
    print("Reports:")
    print(REPORTS_DIR)

    print()

    check_reports()

    (
        solution_targets,
        solution_ranking,
        solution_interactions,
        contribution,
        priority,
        isolation,
        consistency
    ) = load_all_reports()

    factor_plan = build_factor_plan(
        solution_targets,
        contribution,
        priority,
        isolation,
        consistency
    )

    interaction_plan = build_interaction_plan(
        solution_interactions
    )

    experiment_plan = build_experiment_plan(
        factor_plan,
        interaction_plan
    )

    print()
    print(
        "Generando reports..."
    )

    factor_plan_path = (
        REPORTS_DIR
        / OUTPUT_FILES["factor_plan"]
    )

    factor_plan_ranking_path = (
        REPORTS_DIR
        / OUTPUT_FILES["factor_plan_ranking"]
    )

    interaction_plan_path = (
        REPORTS_DIR
        / OUTPUT_FILES["interaction_plan"]
    )

    experiment_plan_path = (
        REPORTS_DIR
        / OUTPUT_FILES["experiment_plan"]
    )

    summary_path = (
        REPORTS_DIR
        / OUTPUT_FILES["summary"]
    )

    write_csv(
        factor_plan_path,
        factor_plan
    )

    print(
        f"[OK] {factor_plan_path}"
    )

    ranking_rows = []

    for row in factor_plan:

        ranking_rows.append({
            "rank":
                row["plan_rank"],

            "factor":
                row["factor"],

            "severity":
                row["severity"],

            "failures":
                row["failures"],

            "contribution_percentage":
                row["contribution_percentage"],

            "priority_score":
                row["priority_score"],

            "isolation_score":
                row["isolation_score"],

            "consistency_score":
                row["consistency_score"],

            "solution_score":
                row["solution_score"],

            "target":
                row["target"]
        })

    write_csv(
        factor_plan_ranking_path,
        ranking_rows
    )

    print(
        f"[OK] {factor_plan_ranking_path}"
    )

    write_csv(
        interaction_plan_path,
        interaction_plan
    )

    print(
        f"[OK] {interaction_plan_path}"
    )

    write_csv(
        experiment_plan_path,
        experiment_plan
    )

    print(
        f"[OK] {experiment_plan_path}"
    )

    summary = generate_summary(
        factor_plan,
        interaction_plan,
        experiment_plan
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(summary)

    print(
        f"[OK] {summary_path}"
    )

    # ========================================================================
    # CONSOLE RESULT
    # ========================================================================

    print_header(
        "RESULTADO PERSON SMALL FAILURE "
        "SOLUTION PLAN V1"
    )

    print(
        f"Factores analizados:      "
        f"{len(factor_plan)}"
    )

    print(
        f"Interacciones:            "
        f"{len(interaction_plan)}"
    )

    print(
        f"Experimentos propuestos:  "
        f"{len(experiment_plan)}"
    )

    print()

    print(
        "TOP SOLUTION PLAN"
    )

    print()

    for row in factor_plan:

        print(
            f"{row['plan_rank']}. "
            f"{row['factor']:<24} "
            f"Failures={row['failures']:6d} "
            f"Contribution="
            f"{row['contribution_percentage']:7.2f}% "
            f"Score="
            f"{row['solution_score']:7.2f}"
        )

        print(
            f"   TARGET: {row['target']}"
        )

        print(
            f"   SEVERITY: {row['severity']}"
        )

    print()

    print(
        "TOP INTERACCIONES"
    )

    print()

    for row in interaction_plan[:15]:

        print(
            f"{row['plan_rank']}. "
            f"{row['factor_a']} + "
            f"{row['factor_b']}"
        )

        print(
            f"   Failures="
            f"{row['failures']:6d} "
            f"Percentage="
            f"{row['percentage_of_total_failures']:7.2f}%"
        )

        print(
            f"   TARGET: "
            f"{row['target']}"
        )

    print()

    print(
        "PLAN EXPERIMENTAL"
    )

    print()

    for experiment in experiment_plan:

        print(
            f"{experiment['experiment_id']} - "
            f"{experiment['name']}"
        )

        print(
            f"   {experiment['objective']}"
        )

    print()

    print(
        "[OK] Reports generados."
    )

    print()

    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )


if __name__ == "__main__":
    main()