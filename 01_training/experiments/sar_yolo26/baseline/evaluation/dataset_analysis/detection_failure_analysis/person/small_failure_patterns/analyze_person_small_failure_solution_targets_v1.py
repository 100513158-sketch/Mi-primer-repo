#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
SAR YOLO26 - PERSON SMALL FAILURE SOLUTION TARGETS ANALYSIS V1
========================================================================

Objetivo
--------
Convertir los resultados acumulados del análisis de PERSON SMALL failures
en objetivos concretos de intervención para mejorar el detector YOLO26.

IMPORTANTE
----------
- Este script NO modifica el dataset.
- Este script NO modifica labels.
- Este script NO modifica imágenes.
- Este script solamente lee reports CSV existentes y genera nuevos reports.

La ruta BASE se determina automáticamente buscando el directorio padre
llamado "baseline". De esta forma el script no depende del directorio
desde el que se ejecute PowerShell.

Estructura esperada:

baseline/
└── evaluation/
    └── dataset_analysis/
        └── detection_failure_analysis/
            └── person/
                └── small_failure_patterns/
                    ├── analyze_person_small_failure_solution_targets_v1.py
                    └── reports/
                        ├── person_small_failure_factor_contribution_v1.csv
                        ├── person_small_failure_factor_ranking_v1.csv
                        ├── person_small_failure_factor_interactions_v1.csv
                        ├── person_small_failure_priority_v1.csv
                        ├── person_small_failure_priority_ranking_v1.csv
                        ├── person_small_failure_priority_interactions_v1.csv
                        ├── person_small_failure_priority_factor_pairs_v1.csv
                        ├── person_small_failure_factor_isolation_v1.csv
                        ├── person_small_failure_factor_isolation_ranking_v1.csv
                        ├── person_small_failure_factor_isolation_interactions_v1.csv
                        ├── person_small_failure_factor_isolation_factor_pairs_v1.csv
                        ├── person_small_failure_factor_isolation_distribution_v1.csv
                        ├── person_small_failure_factor_consistency_v1.csv
                        ├── person_small_failure_factor_consistency_ranking_v1.csv
                        ├── person_small_failure_interaction_consistency_v1.csv
                        └── person_small_failure_consistency_matrix_v1.csv
========================================================================
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

SCRIPT_PATH = Path(__file__).resolve()


# ======================================================================
# RESOLUCIÓN ROBUSTA DE BASE
# ======================================================================

def find_baseline_dir() -> Path:
    """
    Busca automáticamente un directorio padre llamado 'baseline'.

    Esto evita errores como:

        C:\\SARC-Drone\\01_training\\experiments

    cuando realmente necesitamos:

        C:\\SARC-Drone\\01_training\\experiments\\sar_yolo26\\baseline
    """

    candidates = []

    current = SCRIPT_PATH.parent

    for ancestor in [current] + list(current.parents):

        if ancestor.name.lower() == "baseline":
            candidates.append(ancestor)

    for candidate in candidates:

        evaluation_dir = candidate / "evaluation"

        if evaluation_dir.exists():
            return candidate

    # Fallback: intentar localizar "sar_yolo26/baseline"
    for ancestor in [current] + list(current.parents):

        candidate = ancestor / "baseline"

        if (
            candidate.exists()
            and candidate.is_dir()
            and (candidate / "evaluation").exists()
        ):
            return candidate

    raise RuntimeError(
        "\nNo se pudo determinar automáticamente el directorio BASE.\n\n"
        f"Script:\n{SCRIPT_PATH}\n\n"
        "Se esperaba encontrar un directorio padre llamado 'baseline' "
        "que contenga la carpeta 'evaluation'."
    )


BASE_DIR = find_baseline_dir()

REPORTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)


# ======================================================================
# REPORTS DE ENTRADA
# ======================================================================

INPUT_REPORTS = {
    "factor_contribution":
        REPORTS_DIR /
        "person_small_failure_factor_contribution_v1.csv",

    "factor_ranking":
        REPORTS_DIR /
        "person_small_failure_factor_ranking_v1.csv",

    "factor_interactions":
        REPORTS_DIR /
        "person_small_failure_factor_interactions_v1.csv",

    "priority":
        REPORTS_DIR /
        "person_small_failure_priority_v1.csv",

    "priority_ranking":
        REPORTS_DIR /
        "person_small_failure_priority_ranking_v1.csv",

    "priority_interactions":
        REPORTS_DIR /
        "person_small_failure_priority_interactions_v1.csv",

    "priority_pairs":
        REPORTS_DIR /
        "person_small_failure_priority_factor_pairs_v1.csv",

    "isolation":
        REPORTS_DIR /
        "person_small_failure_factor_isolation_v1.csv",

    "isolation_ranking":
        REPORTS_DIR /
        "person_small_failure_factor_isolation_ranking_v1.csv",

    "isolation_interactions":
        REPORTS_DIR /
        "person_small_failure_factor_isolation_interactions_v1.csv",

    "isolation_pairs":
        REPORTS_DIR /
        "person_small_failure_factor_isolation_factor_pairs_v1.csv",

    "isolation_distribution":
        REPORTS_DIR /
        "person_small_failure_factor_isolation_distribution_v1.csv",

    "consistency":
        REPORTS_DIR /
        "person_small_failure_factor_consistency_v1.csv",

    "consistency_ranking":
        REPORTS_DIR /
        "person_small_failure_factor_consistency_ranking_v1.csv",

    "interaction_consistency":
        REPORTS_DIR /
        "person_small_failure_interaction_consistency_v1.csv",

    "consistency_matrix":
        REPORTS_DIR /
        "person_small_failure_consistency_matrix_v1.csv",
}


# ======================================================================
# REPORTS DE SALIDA
# ======================================================================

OUTPUT_FACTOR_TARGETS = (
    REPORTS_DIR /
    "person_small_failure_solution_targets_v1.csv"
)

OUTPUT_FACTOR_RANKING = (
    REPORTS_DIR /
    "person_small_failure_solution_targets_ranking_v1.csv"
)

OUTPUT_INTERACTIONS = (
    REPORTS_DIR /
    "person_small_failure_solution_targets_interactions_v1.csv"
)

OUTPUT_PAIRS = (
    REPORTS_DIR /
    "person_small_failure_solution_targets_factor_pairs_v1.csv"
)

OUTPUT_SUMMARY = (
    REPORTS_DIR /
    "PERSON_SMALL_FAILURE_SOLUTION_TARGETS_V1_SUMMARY.txt"
)


# ======================================================================
# UTILIDADES
# ======================================================================

def normalize_name(value: str) -> str:
    """
    Normaliza nombres de columnas.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def to_float(value, default=0.0) -> float:
    """
    Conversión robusta a float.
    """

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return default


def to_int(value, default=0) -> int:
    """
    Conversión robusta a int.
    """

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    try:
        return int(float(text))
    except ValueError:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_percentage(value: float) -> float:
    return clamp(value, 0.0, 100.0)


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Lee CSV con UTF-8-SIG para soportar BOM de Excel/Windows.
    """

    if not path.exists():
        raise FileNotFoundError(f"No existe:\n{path}")

    if path.stat().st_size == 0:
        raise ValueError(f"El CSV está vacío:\n{path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV sin cabecera:\n{path}"
            )

        fieldnames = [
            str(x).strip()
            for x in reader.fieldnames
        ]

        rows = []

        for row in reader:

            clean_row = {}

            for key, value in row.items():

                if key is None:
                    continue

                clean_row[str(key).strip()] = (
                    "" if value is None else str(value).strip()
                )

            rows.append(clean_row)

    return rows, fieldnames


def normalized_field_map(fieldnames: List[str]) -> Dict[str, str]:
    """
    Devuelve:

        nombre_normalizado -> nombre_original
    """

    result = {}

    for field in fieldnames:
        result[normalize_name(field)] = field

    return result


def get_value(
    row: Dict[str, str],
    candidates: List[str],
    default=None
):
    """
    Obtiene el primer campo existente entre varios candidatos.
    """

    field_map = normalized_field_map(list(row.keys()))

    for candidate in candidates:

        normalized = normalize_name(candidate)

        if normalized in field_map:

            return row[field_map[normalized]]

    return default


# ======================================================================
# COMPROBACIÓN DE REPORTS
# ======================================================================

def check_reports() -> None:

    print("Comprobando reports de entrada...")

    missing = []

    for key, path in INPUT_REPORTS.items():

        if not path.exists():

            print(f"[MISSING] {key}")

            missing.append((key, path))

        elif path.stat().st_size == 0:

            print(f"[EMPTY]   {key}")

            missing.append((key, path))

        else:

            print(f"[OK] {key}")

    if missing:

        print("\n" + "=" * 72)
        print("REPORTS FALTANTES O VACÍOS")
        print("=" * 72)

        for key, path in missing:

            print(f"\n{key}")
            print(path)

        print(
            "\nEl análisis NO puede continuar porque faltan "
            "reports de las fases anteriores."
        )

        raise FileNotFoundError(
            "Faltan uno o más reports necesarios."
        )

    print("\n[OK] Todos los reports de entrada están disponibles.")


# ======================================================================
# CARGA DE FACTORES
# ======================================================================

def load_factor_contribution():

    print("\nCargando contribución de factores...")

    rows, fieldnames = read_csv(
        INPUT_REPORTS["factor_contribution"]
    )

    factors = {}

    for row in rows:

        factor = get_value(
            row,
            ["factor"]
        )

        if not factor:
            continue

        failures = to_int(
            get_value(
                row,
                ["failures", "failure_count"]
            )
        )

        contribution = to_float(
            get_value(
                row,
                [
                    "contribution_percentage",
                    "contribution",
                    "percentage"
                ]
            )
        )

        interaction_count = to_int(
            get_value(
                row,
                [
                    "interaction_count",
                    "interactions"
                ]
            )
        )

        interaction_percentage = to_float(
            get_value(
                row,
                [
                    "interaction_percentage",
                    "interaction_percent"
                ]
            )
        )

        factors[factor] = {
            "factor": factor,
            "failures": failures,
            "contribution_percentage": contribution,
            "interaction_count": interaction_count,
            "interaction_percentage": interaction_percentage,
        }

    print(f"[OK] Factores cargados: {len(factors)}")

    return factors


# ======================================================================
# CARGA PRIORIDAD
# ======================================================================

def load_priority():

    print("\nCargando prioridad de factores...")

    rows, _ = read_csv(
        INPUT_REPORTS["priority_ranking"]
    )

    priority = {}

    for row in rows:

        factor = get_value(
            row,
            ["factor"]
        )

        if not factor:
            continue

        rank = to_int(
            get_value(
                row,
                ["rank", "priority_rank"]
            )
        )

        failures = to_int(
            get_value(
                row,
                ["failures"]
            )
        )

        contribution = to_float(
            get_value(
                row,
                [
                    "contribution_percentage",
                    "contribution"
                ]
            )
        )

        priority_score = to_float(
            get_value(
                row,
                [
                    "priority",
                    "priority_score"
                ]
            )
        )

        priority[factor] = {
            "factor": factor,
            "rank": rank,
            "failures": failures,
            "contribution_percentage": contribution,
            "priority": priority_score,
        }

    print(f"[OK] Priority cargada: {len(priority)}")

    return priority


# ======================================================================
# CARGA AISLAMIENTO
# ======================================================================

def load_isolation():

    print("\nCargando aislamiento de factores...")

    rows, _ = read_csv(
        INPUT_REPORTS["isolation_ranking"]
    )

    isolation = {}

    for row in rows:

        factor = get_value(
            row,
            ["factor"]
        )

        if not factor:
            continue

        rank = to_int(
            get_value(
                row,
                ["rank", "isolation_rank"]
            )
        )

        failures = to_int(
            get_value(
                row,
                ["failures"]
            )
        )

        failure_rate = to_float(
            get_value(
                row,
                [
                    "failure_rate",
                    "failure_rate_percentage"
                ]
            )
        )

        risk_diff = to_float(
            get_value(
                row,
                [
                    "risk_diff",
                    "risk_difference",
                    "risk_diff_pp"
                ]
            )
        )

        isolation_score = to_float(
            get_value(
                row,
                [
                    "priority",
                    "isolation",
                    "isolation_score"
                ]
            )
        )

        isolation[factor] = {
            "factor": factor,
            "rank": rank,
            "failures": failures,
            "failure_rate": failure_rate,
            "risk_diff": risk_diff,
            "isolation_score": isolation_score,
        }

    print(f"[OK] Isolation cargada: {len(isolation)}")

    return isolation


# ======================================================================
# CARGA CONSISTENCIA
# ======================================================================

def load_consistency():

    print("\nCargando consistencia de factores...")

    rows, _ = read_csv(
        INPUT_REPORTS["consistency_ranking"]
    )

    consistency = {}

    for row in rows:

        factor = get_value(
            row,
            ["factor"]
        )

        if not factor:
            continue

        rank = to_int(
            get_value(
                row,
                ["rank", "consistency_rank"]
            )
        )

        consistency_score = to_float(
            get_value(
                row,
                [
                    "consistency",
                    "consistency_score"
                ]
            )
        )

        failures = to_int(
            get_value(
                row,
                ["failures"]
            )
        )

        contribution = to_float(
            get_value(
                row,
                [
                    "contribution_percentage",
                    "contribution"
                ]
            )
        )

        consistency[factor] = {
            "factor": factor,
            "rank": rank,
            "consistency": consistency_score,
            "failures": failures,
            "contribution_percentage": contribution,
        }

    print(f"[OK] Consistency cargada: {len(consistency)}")

    return consistency


# ======================================================================
# CARGA INTERACCIONES
# ======================================================================

def load_interactions():

    print("\nCargando interacciones...")

    rows, _ = read_csv(
        INPUT_REPORTS["factor_interactions"]
    )

    interactions = []

    for row in rows:

        factor_a = get_value(
            row,
            ["factor_a"]
        )

        factor_b = get_value(
            row,
            ["factor_b"]
        )

        interaction = get_value(
            row,
            ["interaction"]
        )

        failures = to_int(
            get_value(
                row,
                ["failures"]
            )
        )

        percentage = to_float(
            get_value(
                row,
                [
                    "percentage_of_total_failures",
                    "percentage",
                    "percentage_of_failures"
                ]
            )
        )

        occurrences = to_int(
            get_value(
                row,
                ["occurrences"]
            )
        )

        if not factor_a or not factor_b:

            continue

        interactions.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "interaction": interaction
                    or f"{factor_a} + {factor_b}",
                "failures": failures,
                "percentage": percentage,
                "occurrences": occurrences,
            }
        )

    print(
        f"[OK] Interacciones cargadas: {len(interactions)}"
    )

    return interactions


# ======================================================================
# MAPA DE SOLUCIONES
# ======================================================================

SOLUTION_MAP = {

    "NO_PREDICTION": {
        "target": "INCREASE_DETECTION_RECALL",
        "area": "DETECTION",
        "problem": "No prediction",
        "solution": (
            "Mejorar sensibilidad del detector para personas pequeñas; "
            "revisar recall, confidence threshold, small-object augmentation "
            "y ejemplos de personas pequeñas sin predicción."
        ),
        "actions": (
            "Revisar confidence threshold; "
            "aumentar ejemplos de small persons; "
            "oversampling selectivo; "
            "small-object augmentation; "
            "revisar imgsz; "
            "evaluar pérdida de clasificación/detección."
        ),
    },

    "DENSE_SCENE": {
        "target": "IMPROVE_DENSE_SCENE_DETECTION",
        "area": "SCENE_COMPLEXITY",
        "problem": "Dense scene",
        "solution": (
            "Mejorar separación y detección de personas en escenas densas "
            "mediante entrenamiento específico para alta densidad."
        ),
        "actions": (
            "Oversampling de escenas densas; "
            "augmentations específicas; "
            "copy-paste controlado; "
            "revisión de NMS; "
            "evaluación de crowd scenes."
        ),
    },

    "EDGE_LOCATION": {
        "target": "IMPROVE_BORDER_AWARE_DETECTION",
        "area": "IMAGE_POSITION",
        "problem": "Edge location",
        "solution": (
            "Mejorar detección de personas próximas a los bordes "
            "de la imagen."
        ),
        "actions": (
            "Crop/padding augmentation; "
            "random crop; "
            "personas parcialmente recortadas; "
            "border-aware augmentation; "
            "revisión de imágenes con personas en extremos."
        ),
    },

    "CLOSE_NEIGHBORS": {
        "target": "IMPROVE_PERSON_SEPARATION",
        "area": "INSTANCE_SEPARATION",
        "problem": "Close neighbors",
        "solution": (
            "Mejorar separación entre personas pequeñas próximas "
            "entre sí."
        ),
        "actions": (
            "Escenas crowd; "
            "augmentations con personas próximas; "
            "copy-paste de grupos; "
            "revisión de NMS; "
            "análisis de IoU entre predicciones vecinas."
        ),
    },

    "EXTREME_SMALL": {
        "target": "IMPROVE_EXTREME_SMALL_OBJECT_DETECTION",
        "area": "SMALL_OBJECTS",
        "problem": "Extreme small",
        "solution": (
            "Mejorar específicamente la representación y detección "
            "de personas extremadamente pequeñas."
        ),
        "actions": (
            "Aumentar resolución de entrada; "
            "tiling/crops; "
            "oversampling; "
            "small-object augmentation; "
            "revisar tamaño mínimo de objetos durante entrenamiento."
        ),
    },

    "LOCALIZATION_ERROR": {
        "target": "IMPROVE_LOCALIZATION",
        "area": "LOCALIZATION",
        "problem": "Localization error",
        "solution": (
            "Mejorar precisión de las bounding boxes en personas pequeñas."
        ),
        "actions": (
            "Revisar box regression; "
            "augmentations geométricas; "
            "analizar IoU; "
            "revisar calidad de labels; "
            "evaluar pérdida de localización."
        ),
    },

    "OCCLUSION": {
        "target": "IMPROVE_OCCLUSION_ROBUSTNESS",
        "area": "OCCLUSION",
        "problem": "Occlusion",
        "solution": (
            "Mejorar robustez frente a personas parcialmente ocultas."
        ),
        "actions": (
            "Copy-paste; "
            "Cutout/Mosaic controlado; "
            "escenas con oclusión; "
            "oversampling de ejemplos ocluidos; "
            "revisión de labels parcialmente visibles."
        ),
    },
}


# ======================================================================
# SOLUCIÓN GENÉRICA
# ======================================================================

def generic_solution(factor: str):

    return {
        "target": f"IMPROVE_{factor}",
        "area": "GENERAL",
        "problem": factor,
        "solution": (
            f"Analizar y reducir específicamente el impacto del factor "
            f"{factor}."
        ),
        "actions": (
            "Oversampling; "
            "augmentation dirigida; "
            "revisión de labels; "
            "evaluación específica del factor."
        ),
    }


# ======================================================================
# SCORE DE SOLUCIÓN
# ======================================================================

def calculate_solution_score(
    contribution: float,
    priority: float,
    isolation: float,
    consistency: float,
) -> float:
    """
    Score compuesto.

    No pretende ser una métrica estadística causal.

    Sirve para ordenar los objetivos de intervención considerando:

        35% contribución
        30% prioridad
        20% aislamiento
        15% consistencia

    Los componentes se normalizan independientemente.
    """

    contribution_n = safe_percentage(contribution)

    priority_n = clamp(priority, 0.0, 100.0)

    isolation_n = clamp(isolation, 0.0, 100.0)

    consistency_n = clamp(consistency, 0.0, 100.0)

    score = (
        contribution_n * 0.35
        + priority_n * 0.30
        + isolation_n * 0.20
        + consistency_n * 0.15
    )

    return round(score, 4)


# ======================================================================
# NORMALIZACIÓN DE SCORES
# ======================================================================

def normalize_values(values: Dict[str, float]) -> Dict[str, float]:

    if not values:
        return {}

    maximum = max(values.values())

    if maximum <= 0:
        return {
            key: 0.0
            for key in values
        }

    return {
        key: (value / maximum) * 100.0
        for key, value in values.items()
    }


# ======================================================================
# SOLUTION TARGETS DE FACTORES
# ======================================================================

def calculate_factor_targets(
    factors,
    priority,
    isolation,
    consistency
):

    print("\nCalculando solution targets de factores...")

    priority_values = {
        factor: data.get("priority", 0.0)
        for factor, data in priority.items()
    }

    isolation_values = {
        factor: abs(data.get("isolation_score", 0.0))
        for factor, data in isolation.items()
    }

    consistency_values = {
        factor: data.get("consistency", 0.0)
        for factor, data in consistency.items()
    }

    priority_norm = normalize_values(priority_values)

    isolation_norm = normalize_values(isolation_values)

    consistency_norm = normalize_values(
        consistency_values
    )

    results = []

    all_factors = set(factors)

    for factor in all_factors:

        contribution_data = factors.get(
            factor,
            {}
        )

        priority_data = priority.get(
            factor,
            {}
        )

        isolation_data = isolation.get(
            factor,
            {}
        )

        consistency_data = consistency.get(
            factor,
            {}
        )

        failures = to_int(
            contribution_data.get("failures", 0)
        )

        contribution = to_float(
            contribution_data.get(
                "contribution_percentage",
                0.0
            )
        )

        priority_score = to_float(
            priority_data.get(
                "priority",
                0.0
            )
        )

        isolation_score = abs(
            to_float(
                isolation_data.get(
                    "isolation_score",
                    0.0
                )
            )
        )

        consistency_score = to_float(
            consistency_data.get(
                "consistency",
                0.0
            )
        )

        score = calculate_solution_score(
            contribution,
            priority_norm.get(
                factor,
                0.0
            ),
            isolation_norm.get(
                factor,
                0.0
            ),
            consistency_norm.get(
                factor,
                0.0
            ),
        )

        solution = SOLUTION_MAP.get(
            factor,
            generic_solution(factor)
        )

        results.append(
            {
                "factor": factor,
                "failures": failures,
                "contribution_percentage": round(
                    contribution,
                    6
                ),
                "priority_score": round(
                    priority_score,
                    6
                ),
                "priority_normalized": round(
                    priority_norm.get(
                        factor,
                        0.0
                    ),
                    6
                ),
                "isolation_score": round(
                    isolation_score,
                    6
                ),
                "isolation_normalized": round(
                    isolation_norm.get(
                        factor,
                        0.0
                    ),
                    6
                ),
                "consistency_score": round(
                    consistency_score,
                    6
                ),
                "consistency_normalized": round(
                    consistency_norm.get(
                        factor,
                        0.0
                    ),
                    6
                ),
                "solution_score": score,
                "solution_target": solution["target"],
                "solution_area": solution["area"],
                "problem": solution["problem"],
                "recommended_solution": solution["solution"],
                "recommended_actions": solution["actions"],
            }
        )

    results.sort(
        key=lambda x: (
            x["solution_score"],
            x["contribution_percentage"],
            x["failures"],
        ),
        reverse=True
    )

    for index, row in enumerate(
        results,
        start=1
    ):
        row["solution_rank"] = index

    print(
        f"[OK] Solution targets calculados: {len(results)}"
    )

    return results


# ======================================================================
# SOLUTION TARGETS DE INTERACCIONES
# ======================================================================

def calculate_interaction_targets(
    interactions,
    factor_targets
):

    print("\nCalculando solution targets de interacciones...")

    factor_score_map = {
        row["factor"]: row["solution_score"]
        for row in factor_targets
    }

    results = []

    for interaction in interactions:

        factor_a = interaction["factor_a"]

        factor_b = interaction["factor_b"]

        failures = interaction["failures"]

        percentage = interaction["percentage"]

        score_a = factor_score_map.get(
            factor_a,
            0.0
        )

        score_b = factor_score_map.get(
            factor_b,
            0.0
        )

        interaction_score = (
            percentage * 0.60
            + ((score_a + score_b) / 2.0) * 0.40
        )

        solution_a = SOLUTION_MAP.get(
            factor_a,
            generic_solution(factor_a)
        )

        solution_b = SOLUTION_MAP.get(
            factor_b,
            generic_solution(factor_b)
        )

        combined_target = (
            f"{solution_a['target']} + "
            f"{solution_b['target']}"
        )

        combined_actions = (
            f"{solution_a['actions']} "
            f"{solution_b['actions']}"
        )

        results.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "interaction": interaction["interaction"],
                "failures": failures,
                "percentage_of_total_failures": round(
                    percentage,
                    6
                ),
                "factor_a_solution_score": round(
                    score_a,
                    6
                ),
                "factor_b_solution_score": round(
                    score_b,
                    6
                ),
                "interaction_solution_score": round(
                    interaction_score,
                    6
                ),
                "solution_target": combined_target,
                "recommended_actions": combined_actions,
            }
        )

    results.sort(
        key=lambda x: (
            x["interaction_solution_score"],
            x["failures"],
        ),
        reverse=True
    )

    for index, row in enumerate(
        results,
        start=1
    ):
        row["solution_rank"] = index

    print(
        f"[OK] Interacciones priorizadas: {len(results)}"
    )

    return results


# ======================================================================
# PARES DE FACTORES
# ======================================================================

def calculate_factor_pairs(
    factor_targets,
    interaction_targets
):

    print("\nCalculando pares de factores...")

    target_map = {
        row["factor"]: row
        for row in factor_targets
    }

    results = []

    for interaction in interaction_targets:

        factor_a = interaction["factor_a"]

        factor_b = interaction["factor_b"]

        a = target_map.get(
            factor_a,
            {}
        )

        b = target_map.get(
            factor_b,
            {}
        )

        score_a = to_float(
            a.get(
                "solution_score",
                0.0
            )
        )

        score_b = to_float(
            b.get(
                "solution_score",
                0.0
            )
        )

        contribution_a = to_float(
            a.get(
                "contribution_percentage",
                0.0
            )
        )

        contribution_b = to_float(
            b.get(
                "contribution_percentage",
                0.0
            )
        )

        pair_score = (
            interaction["interaction_solution_score"] * 0.50
            + ((score_a + score_b) / 2.0) * 0.30
            + ((contribution_a + contribution_b) / 2.0) * 0.20
        )

        results.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "interaction": interaction["interaction"],
                "failures": interaction["failures"],
                "percentage_of_total_failures":
                    interaction[
                        "percentage_of_total_failures"
                    ],
                "factor_a_score": round(
                    score_a,
                    6
                ),
                "factor_b_score": round(
                    score_b,
                    6
                ),
                "pair_solution_score": round(
                    pair_score,
                    6
                ),
                "recommended_target":
                    interaction["solution_target"],
            }
        )

    results.sort(
        key=lambda x: (
            x["pair_solution_score"],
            x["failures"],
        ),
        reverse=True
    )

    for index, row in enumerate(
        results,
        start=1
    ):
        row["pair_rank"] = index

    print(
        f"[OK] Pares calculados: {len(results)}"
    )

    return results


# ======================================================================
# ESCRITURA CSV
# ======================================================================

def write_csv(
    path: Path,
    rows: List[Dict]
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        raise ValueError(
            f"No hay datos para escribir:\n{path}"
        )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ======================================================================
# SUMMARY
# ======================================================================

def generate_summary(
    factor_targets,
    interaction_targets,
    pair_targets
):

    print("\nGenerando summary...")

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "SAR YOLO26 - PERSON SMALL FAILURE SOLUTION TARGETS V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"BASE: {BASE_DIR}"
    )

    lines.append(
        f"REPORTS: {REPORTS_DIR}"
    )

    lines.append("")

    lines.append(
        f"Factores analizados:      {len(factor_targets)}"
    )

    lines.append(
        f"Interacciones analizadas: {len(interaction_targets)}"
    )

    lines.append(
        f"Pares de factores:        {len(pair_targets)}"
    )

    lines.append("")

    lines.append(
        "TOP FACTORES - SOLUTION TARGETS"
    )

    lines.append(
        "-" * 72
    )

    for row in factor_targets[:10]:

        lines.append(
            f"{row['solution_rank']:>2}. "
            f"{row['factor']:<25} "
            f"Failures={row['failures']:>6} "
            f"Contribution="
            f"{row['contribution_percentage']:>7.2f}% "
            f"Score="
            f"{row['solution_score']:>7.2f}"
        )

        lines.append(
            f"    TARGET: {row['solution_target']}"
        )

        lines.append(
            f"    AREA:   {row['solution_area']}"
        )

        lines.append(
            f"    SOLUTION: {row['recommended_solution']}"
        )

        lines.append(
            f"    ACTIONS: {row['recommended_actions']}"
        )

        lines.append("")

    lines.append(
        "TOP INTERACCIONES - SOLUTION TARGETS"
    )

    lines.append(
        "-" * 72
    )

    for row in interaction_targets[:15]:

        lines.append(
            f"{row['solution_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures={row['failures']:>6} "
            f"Percentage="
            f"{row['percentage_of_total_failures']:>7.2f}% "
            f"Score="
            f"{row['interaction_solution_score']:>7.2f}"
        )

        lines.append(
            f"    TARGET: {row['solution_target']}"
        )

        lines.append("")

    lines.append(
        "TOP PARES DE FACTORES"
    )

    lines.append(
        "-" * 72
    )

    for row in pair_targets[:15]:

        lines.append(
            f"{row['pair_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures={row['failures']:>6} "
            f"Score="
            f"{row['pair_solution_score']:>7.2f}"
        )

        lines.append(
            f"    TARGET: {row['recommended_target']}"
        )

        lines.append("")

    lines.append(
        "=" * 72
    )

    lines.append(
        "INTERPRETACION"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "Los solution targets no representan causalidad estadística."
    )

    lines.append(
        "Se utilizan como mecanismo de priorización de acciones "
        "de mejora del detector."
    )

    lines.append("")

    lines.append(
        "La prioridad debe centrarse inicialmente en los factores "
        "con mayor combinación de contribución, prioridad, "
        "aislamiento y consistencia."
    )

    lines.append("")

    lines.append(
        "Para este análisis los principales objetivos esperados son:"
    )

    lines.append(
        "1. Mejorar recall de personas pequeñas sin predicción."
    )

    lines.append(
        "2. Mejorar detección en escenas densas."
    )

    lines.append(
        "3. Mejorar separación entre personas próximas."
    )

    lines.append(
        "4. Mejorar detección cerca de los bordes."
    )

    lines.append(
        "5. Mejorar detección de objetos extremadamente pequeños."
    )

    lines.append(
        "6. Mejorar localización."
    )

    lines.append(
        "7. Mejorar robustez frente a oclusión."
    )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    lines.append("")

    OUTPUT_SUMMARY.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"[OK] {OUTPUT_SUMMARY}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE "
        "SOLUTION TARGETS ANALYSIS V1"
    )
    print("=" * 72)

    print()
    print("SCRIPT:")
    print(SCRIPT_PATH)

    print()
    print("BASE:")
    print(BASE_DIR)

    print()
    print("Reports:")
    print(REPORTS_DIR)

    print()

    # --------------------------------------------------------------
    # Comprobación de estructura
    # --------------------------------------------------------------

    if BASE_DIR.name.lower() != "baseline":

        raise RuntimeError(
            "La BASE detectada no termina en 'baseline':\n"
            f"{BASE_DIR}"
        )

    if not (
        BASE_DIR / "evaluation"
    ).exists():

        raise RuntimeError(
            "La BASE no contiene evaluation:\n"
            f"{BASE_DIR}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Comprobar reports
    # --------------------------------------------------------------

    check_reports()

    # --------------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------------

    factors = load_factor_contribution()

    priority = load_priority()

    isolation = load_isolation()

    consistency = load_consistency()

    interactions = load_interactions()

    # --------------------------------------------------------------
    # Comprobar coherencia
    # --------------------------------------------------------------

    print("\nComprobando coherencia de factores...")

    factor_sets = [
        set(factors.keys()),
        set(priority.keys()),
        set(isolation.keys()),
        set(consistency.keys()),
    ]

    common_factors = set.intersection(
        *factor_sets
    )

    if not common_factors:

        raise ValueError(
            "No existe ningún factor común entre "
            "contribution, priority, isolation y consistency."
        )

    print(
        f"[OK] Factores comunes: "
        f"{len(common_factors)}"
    )

    missing_from_any = set.union(
        *factor_sets
    ) - common_factors

    if missing_from_any:

        print(
            "[WARN] Factores no presentes en todos "
            "los reports:"
        )

        for factor in sorted(
            missing_from_any
        ):

            print(
                f"       {factor}"
            )

    # --------------------------------------------------------------
    # Calcular factores
    # --------------------------------------------------------------

    factor_targets = calculate_factor_targets(
        factors,
        priority,
        isolation,
        consistency,
    )

    # --------------------------------------------------------------
    # Calcular interacciones
    # --------------------------------------------------------------

    interaction_targets = calculate_interaction_targets(
        interactions,
        factor_targets,
    )

    # --------------------------------------------------------------
    # Calcular pares
    # --------------------------------------------------------------

    pair_targets = calculate_factor_pairs(
        factor_targets,
        interaction_targets,
    )

    # --------------------------------------------------------------
    # Generar reports
    # --------------------------------------------------------------

    print("\nGenerando reports...")

    write_csv(
        OUTPUT_FACTOR_TARGETS,
        factor_targets,
    )

    print(
        f"[OK] {OUTPUT_FACTOR_TARGETS}"
    )

    write_csv(
        OUTPUT_FACTOR_RANKING,
        factor_targets,
    )

    print(
        f"[OK] {OUTPUT_FACTOR_RANKING}"
    )

    write_csv(
        OUTPUT_INTERACTIONS,
        interaction_targets,
    )

    print(
        f"[OK] {OUTPUT_INTERACTIONS}"
    )

    write_csv(
        OUTPUT_PAIRS,
        pair_targets,
    )

    print(
        f"[OK] {OUTPUT_PAIRS}"
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    generate_summary(
        factor_targets,
        interaction_targets,
        pair_targets,
    )

    # --------------------------------------------------------------
    # Resultado por consola
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON SMALL FAILURE SOLUTION TARGETS V1"
    )
    print("=" * 72)

    print()

    print(
        f"Factores analizados:      "
        f"{len(factor_targets)}"
    )

    print(
        f"Interacciones analizadas: "
        f"{len(interaction_targets)}"
    )

    print(
        f"Pares de factores:        "
        f"{len(pair_targets)}"
    )

    print()
    print(
        "TOP SOLUTION TARGETS"
    )

    print()

    for row in factor_targets[:7]:

        print(
            f"{row['solution_rank']}. "
            f"{row['factor']:<25} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Contribution="
            f"{row['contribution_percentage']:>7.2f}% "
            f"Score="
            f"{row['solution_score']:>7.2f}"
        )

        print(
            f"   TARGET: "
            f"{row['solution_target']}"
        )

    print()

    print(
        "TOP INTERACCIONES"
    )

    print()

    for row in interaction_targets[:15]:

        print(
            f"{row['solution_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Percentage="
            f"{row['percentage_of_total_failures']:>7.2f}% "
            f"Score="
            f"{row['interaction_solution_score']:>7.2f}"
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


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nProceso cancelado por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("ERROR")
        print("=" * 72)
        print()
        print(str(exc))
        print()

        sys.exit(1)