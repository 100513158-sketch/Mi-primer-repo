from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations


# ============================================================================
# SAR YOLO26 - PERSON SMALL FAILURE FACTOR ISOLATION V1
# ============================================================================

SCRIPT_VERSION = "V1"

BASELINE_DIR = (
    Path(__file__).resolve().parents[5]
)

# ---------------------------------------------------------------------------
# Rutas reales del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve()

# Buscar baseline subiendo hasta encontrar evaluation
BASELINE_ROOT = None

for parent in PROJECT_ROOT.parents:
    if (
        (parent / "evaluation").exists()
        and (parent / "training").exists()
    ):
        BASELINE_ROOT = parent
        break

if BASELINE_ROOT is None:
    # Fallback: estructura conocida
    BASELINE_ROOT = (
        Path(r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline")
    )


INPUT_CSV = (
    BASELINE_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_patterns_v1"
    / "reports"
    / "person_small_failure_patterns_objects_v1.csv"
)

OUTPUT_DIR = (
    BASELINE_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

IOU_THRESHOLD = 0.50

# Small person
SMALL_AREA_THRESHOLD = 256.0

# Extreme small
EXTREME_SMALL_AREA_THRESHOLD = 16.0

# ---------------------------------------------------------------------------
# Factores
# ---------------------------------------------------------------------------

FACTORS = [
    "NO_PREDICTION",
    "DENSE_SCENE",
    "EDGE_LOCATION",
    "CLOSE_NEIGHBORS",
    "EXTREME_SMALL",
    "LOCALIZATION_ERROR",
    "OCCLUSION",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def normalize(value):
    if value is None:
        return ""

    return str(value).strip()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentage(value, total):
    if total <= 0:
        return 0.0

    return (value / total) * 100.0


def csv_write(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================================
# DETECCIÓN DE COLUMNAS
# ============================================================================

def detect_required_columns(fieldnames):

    required = [
        "image",
        "image_width",
        "image_height",
        "person_gt_index",
        "area",
        "size_bucket",
        "density_count",
        "density_bucket",
        "location",
        "border",
        "occlusion_bucket",
        "nearest_person_proximity",
        "best_prediction_iou",
        "prediction_relation",
        "dominant_pattern",
    ]

    missing = [
        column
        for column in required
        if column not in fieldnames
    ]

    if missing:
        raise ValueError(
            "Faltan columnas requeridas en el CSV:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


# ============================================================================
# CARGA CSV
# ============================================================================

def load_objects():

    print("Comprobando CSV de objetos...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV:\n{INPUT_CSV}"
        )

    print("[OK] CSV encontrado.")

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []

        detect_required_columns(fieldnames)

        rows = list(reader)

    print(f"[OK] Objetos cargados: {len(rows):,}")

    return rows, fieldnames


# ============================================================================
# ESTADO TP / FN
# ============================================================================

def is_tp(row):

    iou = safe_float(
        row.get("best_prediction_iou")
    )

    return iou >= IOU_THRESHOLD


def is_fn(row):

    return not is_tp(row)


# ============================================================================
# FACTORES
# ============================================================================

def factor_no_prediction(row):

    iou = safe_float(
        row.get("best_prediction_iou")
    )

    relation = normalize(
        row.get("prediction_relation")
    ).upper()

    return (
        iou <= 0.0
        or relation in {
            "NO_PREDICTION",
            "NO_PREDICTION_NEAR",
            "NO_PREDICTION_FAR",
        }
    )


def factor_dense_scene(row):

    density_count = safe_float(
        row.get("density_count")
    )

    density_bucket = normalize(
        row.get("density_bucket")
    ).upper()

    # La clasificación principal se basa en density_count.
    #
    # En este análisis:
    #   <25  -> no dense
    #   >=25 -> dense
    #
    # También se acepta una etiqueta explícita si existiera.

    if density_count >= 25:
        return True

    return density_bucket in {
        "25-50",
        "50-100",
        ">100",
        "DENSE",
        "VERY_DENSE",
        "EXTREME_DENSE",
    }


def factor_edge_location(row):

    border = normalize(
        row.get("border")
    ).upper()

    location = normalize(
        row.get("location")
    ).upper()

    if border and border != "INTERIOR":
        return True

    edge_locations = {
        "TOP_LEFT",
        "TOP_CENTER",
        "TOP_RIGHT",
        "CENTER_LEFT",
        "CENTER_RIGHT",
        "BOTTOM_LEFT",
        "BOTTOM_CENTER",
        "BOTTOM_RIGHT",
    }

    return location in edge_locations


def factor_close_neighbors(row):

    proximity = normalize(
        row.get("nearest_person_proximity")
    ).upper()

    return proximity in {
        "NEAR",
        "VERY_NEAR",
        "CLOSE",
        "VERY_CLOSE",
    }


def factor_extreme_small(row):

    area = safe_float(
        row.get("area")
    )

    return area < EXTREME_SMALL_AREA_THRESHOLD


def factor_localization_error(row):

    iou = safe_float(
        row.get("best_prediction_iou")
    )

    # Existe predicción, pero no alcanza IoU 0.50.
    return (
        iou > 0.0
        and iou < IOU_THRESHOLD
    )


def factor_occlusion(row):

    occlusion = normalize(
        row.get("occlusion_bucket")
    ).upper()

    return occlusion not in {
        "",
        "NO_OVERLAP",
        "NONE",
        "NO_OCCLUSION",
        "VISIBLE",
        "FULLY_VISIBLE",
    }


def detect_factors(row):

    factors = {}

    factors["NO_PREDICTION"] = (
        factor_no_prediction(row)
    )

    factors["DENSE_SCENE"] = (
        factor_dense_scene(row)
    )

    factors["EDGE_LOCATION"] = (
        factor_edge_location(row)
    )

    factors["CLOSE_NEIGHBORS"] = (
        factor_close_neighbors(row)
    )

    factors["EXTREME_SMALL"] = (
        factor_extreme_small(row)
    )

    factors["LOCALIZATION_ERROR"] = (
        factor_localization_error(row)
    )

    factors["OCCLUSION"] = (
        factor_occlusion(row)
    )

    return factors


# ============================================================================
# AISLAMIENTO DE FACTORES
# ============================================================================

def calculate_factor_isolation(rows):

    results = []

    total_objects = len(rows)

    for factor in FACTORS:

        factor_rows = []

        for row in rows:

            factors = detect_factors(row)

            if factors[factor]:
                factor_rows.append(row)

        total_factor = len(factor_rows)

        tp = sum(
            1
            for row in factor_rows
            if is_tp(row)
        )

        fn = sum(
            1
            for row in factor_rows
            if is_fn(row)
        )

        recall = (
            tp / total_factor
            if total_factor > 0
            else 0.0
        )

        failure_rate = (
            fn / total_factor
            if total_factor > 0
            else 0.0
        )

        # --------------------------------------------------------------------
        # Factor ausente
        # --------------------------------------------------------------------

        without_rows = []

        for row in rows:

            factors = detect_factors(row)

            if not factors[factor]:
                without_rows.append(row)

        without_total = len(without_rows)

        without_tp = sum(
            1
            for row in without_rows
            if is_tp(row)
        )

        without_fn = sum(
            1
            for row in without_rows
            if is_fn(row)
        )

        without_recall = (
            without_tp / without_total
            if without_total > 0
            else 0.0
        )

        without_failure_rate = (
            without_fn / without_total
            if without_total > 0
            else 0.0
        )

        # --------------------------------------------------------------------
        # Diferencia de riesgo
        # --------------------------------------------------------------------

        risk_difference = (
            failure_rate
            - without_failure_rate
        )

        recall_difference = (
            recall
            - without_recall
        )

        results.append(
            {
                "factor": factor,
                "objects_with_factor": total_factor,
                "failures_with_factor": fn,
                "tp_with_factor": tp,
                "recall_with_factor": round(
                    recall,
                    6,
                ),
                "failure_rate_with_factor": round(
                    failure_rate,
                    6,
                ),
                "objects_without_factor": without_total,
                "failures_without_factor": without_fn,
                "tp_without_factor": without_tp,
                "recall_without_factor": round(
                    without_recall,
                    6,
                ),
                "failure_rate_without_factor": round(
                    without_failure_rate,
                    6,
                ),
                "risk_difference": round(
                    risk_difference,
                    6,
                ),
                "recall_difference": round(
                    recall_difference,
                    6,
                ),
                "factor_percentage": round(
                    percentage(
                        total_factor,
                        total_objects,
                    ),
                    6,
                ),
                "failure_contribution_percentage": round(
                    percentage(
                        fn,
                        sum(
                            1
                            for r in rows
                            if is_fn(r)
                        ),
                    ),
                    6,
                ),
            }
        )

    return results


# ============================================================================
# RANKING DE AISLAMIENTO
# ============================================================================

def calculate_isolation_ranking(results):

    ranked = []

    for row in results:

        risk = safe_float(
            row["risk_difference"]
        )

        contribution = safe_float(
            row["failure_contribution_percentage"]
        )

        factor_percentage = safe_float(
            row["factor_percentage"]
        )

        # --------------------------------------------------------------------
        # Priority score
        #
        # Combina:
        #   50% riesgo diferencial
        #   30% contribución al fallo
        #   20% presencia del factor
        # --------------------------------------------------------------------

        risk_score = max(
            0.0,
            min(
                1.0,
                risk,
            ),
        )

        contribution_score = (
            max(
                0.0,
                min(
                    100.0,
                    contribution,
                ),
            )
            / 100.0
        )

        presence_score = (
            max(
                0.0,
                min(
                    100.0,
                    factor_percentage,
                ),
            )
            / 100.0
        )

        priority = (
            50.0 * risk_score
            + 30.0 * contribution_score
            + 20.0 * presence_score
        )

        new_row = dict(row)

        new_row["isolation_priority"] = round(
            priority,
            6,
        )

        ranked.append(new_row)

    ranked.sort(
        key=lambda x: (
            safe_float(
                x["isolation_priority"]
            ),
            safe_float(
                x["risk_difference"]
            ),
            safe_float(
                x["failure_contribution_percentage"]
            ),
        ),
        reverse=True,
    )

    for index, row in enumerate(
        ranked,
        start=1,
    ):
        row["rank"] = index

    return ranked


# ============================================================================
# INTERACCIONES AISLADAS
# ============================================================================

def calculate_factor_interactions(rows):

    total_failures = sum(
        1
        for row in rows
        if is_fn(row)
    )

    results = []

    for factor_a, factor_b in combinations(
        FACTORS,
        2,
    ):

        failures = 0
        total = 0

        for row in rows:

            factors = detect_factors(row)

            if (
                factors[factor_a]
                and factors[factor_b]
            ):

                total += 1

                if is_fn(row):
                    failures += 1

        if total == 0:
            continue

        tp = total - failures

        recall = (
            tp / total
            if total > 0
            else 0.0
        )

        failure_rate = (
            failures / total
            if total > 0
            else 0.0
        )

        contribution = (
            percentage(
                failures,
                total_failures,
            )
            if total_failures > 0
            else 0.0
        )

        results.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "objects": total,
                "failures": failures,
                "tp": tp,
                "recall": round(
                    recall,
                    6,
                ),
                "failure_rate": round(
                    failure_rate,
                    6,
                ),
                "failure_contribution_percentage": round(
                    contribution,
                    6,
                ),
            }
        )

    results.sort(
        key=lambda x: (
            safe_float(
                x["failures"]
            ),
            safe_float(
                x["failure_rate"]
            ),
        ),
        reverse=True,
    )

    for index, row in enumerate(
        results,
        start=1,
    ):
        row["rank"] = index

    return results


# ============================================================================
# MATRIZ DE FACTORES
# ============================================================================

def calculate_factor_pairs(rows):

    results = []

    for factor_a, factor_b in combinations(
        FACTORS,
        2,
    ):

        with_both = []
        with_a_only = []
        with_b_only = []
        with_neither = []

        for row in rows:

            factors = detect_factors(row)

            a = factors[factor_a]
            b = factors[factor_b]

            if a and b:
                with_both.append(row)

            elif a and not b:
                with_a_only.append(row)

            elif not a and b:
                with_b_only.append(row)

            else:
                with_neither.append(row)

        def failure_rate(data):

            if not data:
                return 0.0

            return sum(
                1
                for r in data
                if is_fn(r)
            ) / len(data)

        results.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,

                "both_objects": len(
                    with_both
                ),

                "both_failures": sum(
                    1
                    for r in with_both
                    if is_fn(r)
                ),

                "both_failure_rate": round(
                    failure_rate(
                        with_both
                    ),
                    6,
                ),

                "a_only_objects": len(
                    with_a_only
                ),

                "a_only_failures": sum(
                    1
                    for r in with_a_only
                    if is_fn(r)
                ),

                "a_only_failure_rate": round(
                    failure_rate(
                        with_a_only
                    ),
                    6,
                ),

                "b_only_objects": len(
                    with_b_only
                ),

                "b_only_failures": sum(
                    1
                    for r in with_b_only
                    if is_fn(r)
                ),

                "b_only_failure_rate": round(
                    failure_rate(
                        with_b_only
                    ),
                    6,
                ),

                "neither_objects": len(
                    with_neither
                ),

                "neither_failures": sum(
                    1
                    for r in with_neither
                    if is_fn(r)
                ),

                "neither_failure_rate": round(
                    failure_rate(
                        with_neither
                    ),
                    6,
                ),
            }
        )

    return results


# ============================================================================
# DISTRIBUCIÓN DE FACTORES
# ============================================================================

def calculate_factor_distribution(rows):

    total_failures = sum(
        1
        for row in rows
        if is_fn(row)
    )

    results = []

    for factor in FACTORS:

        count_all = 0
        count_failures = 0

        for row in rows:

            factors = detect_factors(row)

            if factors[factor]:

                count_all += 1

                if is_fn(row):
                    count_failures += 1

        results.append(
            {
                "factor": factor,
                "objects": count_all,
                "failures": count_failures,
                "tp": count_all - count_failures,
                "object_percentage": round(
                    percentage(
                        count_all,
                        len(rows),
                    ),
                    6,
                ),
                "failure_percentage": round(
                    percentage(
                        count_failures,
                        total_failures,
                    ),
                    6,
                ),
            }
        )

    results.sort(
        key=lambda x: x["failures"],
        reverse=True,
    )

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(
    rows,
    isolation,
    ranking,
    interactions,
    pairs,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "PERSON_SMALL_FAILURE_FACTOR_ISOLATION_V1_SUMMARY.txt"
    )

    total_objects = len(rows)

    total_failures = sum(
        1
        for row in rows
        if is_fn(row)
    )

    total_tp = (
        total_objects
        - total_failures
    )

    recall = (
        total_tp / total_objects
        if total_objects > 0
        else 0.0
    )

    lines = []

    lines.append("=" * 72)
    lines.append(
        "# SAR YOLO26 - PERSON SMALL FAILURE FACTOR ISOLATION V1"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"Objetos analizados:       {total_objects:,}"
    )
    lines.append(
        f"PERSON TP:                {total_tp:,}"
    )
    lines.append(
        f"PERSON FN:                {total_failures:,}"
    )
    lines.append(
        f"PERSON Recall:            {recall:.4f}"
    )
    lines.append("")
    lines.append(
        f"Factores analizados:      {len(isolation)}"
    )
    lines.append(
        f"Pares de factores:        {len(pairs)}"
    )
    lines.append(
        f"Interacciones:            {len(interactions)}"
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "TOP FACTORES POR AISLAMIENTO"
    )
    lines.append("")

    for row in ranking:

        lines.append(
            f"{row['rank']:>2}. "
            f"{row['factor']:<25} "
            f"Failures={row['failures_with_factor']:>6,} "
            f"FailureRate="
            f"{row['failure_rate_with_factor'] * 100:>6.2f}% "
            f"RiskDiff="
            f"{row['risk_difference'] * 100:>6.2f}pp "
            f"Priority="
            f"{row['isolation_priority']:>6.2f}"
        )

    lines.append("")
    lines.append(
        "=" * 72
    )
    lines.append("")
    lines.append(
        "TOP INTERACCIONES"
    )
    lines.append("")

    for row in interactions[:20]:

        interaction_name = (
            f"{row['factor_a']} + "
            f"{row['factor_b']}"
        )

        lines.append(
            f"{row['rank']:>2}. "
            f"{interaction_name:<55} "
            f"Failures="
            f"{row['failures']:>6,} "
            f"FailureRate="
            f"{row['failure_rate'] * 100:>6.2f}%"
        )

    lines.append("")
    lines.append(
        "=" * 72
    )
    lines.append("")
    lines.append(
        "INTERPRETACIÓN"
    )
    lines.append("")
    lines.append(
        "El aislamiento compara la tasa de fallo cuando un factor "
        "está presente frente a la tasa de fallo cuando el factor "
        "está ausente."
    )
    lines.append("")
    lines.append(
        "RiskDiff = FailureRate(con factor) - "
        "FailureRate(sin factor)."
    )
    lines.append("")
    lines.append(
        "Un RiskDiff positivo indica que la presencia del factor "
        "está asociada a una mayor tasa de fallo."
    )
    lines.append("")
    lines.append(
        "La prioridad combina riesgo diferencial, contribución "
        "al conjunto de fallos y presencia del factor."
    )
    lines.append("")
    lines.append(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE FACTOR ISOLATION V1"
    )
    print("=" * 72)
    print()

    print("Input:")
    print(INPUT_CSV)
    print()

    print("Output:")
    print(OUTPUT_DIR)
    print()

    rows, fieldnames = load_objects()

    print()
    print("Columnas detectadas:")

    for column in fieldnames:
        print(
            f"  {column}"
        )

    print()

    # ------------------------------------------------------------------------
    # TP / FN
    # ------------------------------------------------------------------------

    total_failures = sum(
        1
        for row in rows
        if is_fn(row)
    )

    total_tp = (
        len(rows)
        - total_failures
    )

    print(
        f"[OK] TP detectados: {total_tp:,}"
    )

    print(
        f"[OK] FN detectados: {total_failures:,}"
    )

    print()

    # ------------------------------------------------------------------------
    # Aislamiento
    # ------------------------------------------------------------------------

    print(
        "Calculando aislamiento de factores..."
    )

    isolation = calculate_factor_isolation(
        rows
    )

    print(
        f"[OK] Factores aislados: {len(isolation)}"
    )

    print()

    # ------------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------------

    print(
        "Calculando ranking de aislamiento..."
    )

    ranking = calculate_isolation_ranking(
        isolation
    )

    print(
        f"[OK] Factores priorizados: {len(ranking)}"
    )

    print()

    # ------------------------------------------------------------------------
    # Interacciones
    # ------------------------------------------------------------------------

    print(
        "Calculando interacciones..."
    )

    interactions = calculate_factor_interactions(
        rows
    )

    print(
        f"[OK] Interacciones calculadas: "
        f"{len(interactions)}"
    )

    print()

    # ------------------------------------------------------------------------
    # Pares
    # ------------------------------------------------------------------------

    print(
        "Calculando pares de factores..."
    )

    pairs = calculate_factor_pairs(
        rows
    )

    print(
        f"[OK] Pares calculados: {len(pairs)}"
    )

    print()

    # ------------------------------------------------------------------------
    # Distribución
    # ------------------------------------------------------------------------

    distribution = calculate_factor_distribution(
        rows
    )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    isolation_csv = (
        OUTPUT_DIR
        / "person_small_failure_factor_isolation_v1.csv"
    )

    ranking_csv = (
        OUTPUT_DIR
        / "person_small_failure_factor_isolation_ranking_v1.csv"
    )

    interactions_csv = (
        OUTPUT_DIR
        / "person_small_failure_factor_isolation_interactions_v1.csv"
    )

    pairs_csv = (
        OUTPUT_DIR
        / "person_small_failure_factor_isolation_factor_pairs_v1.csv"
    )

    distribution_csv = (
        OUTPUT_DIR
        / "person_small_failure_factor_isolation_distribution_v1.csv"
    )

    # ------------------------------------------------------------------------
    # CSV 1
    # ------------------------------------------------------------------------

    csv_write(
        isolation_csv,
        [
            "factor",
            "objects_with_factor",
            "failures_with_factor",
            "tp_with_factor",
            "recall_with_factor",
            "failure_rate_with_factor",
            "objects_without_factor",
            "failures_without_factor",
            "tp_without_factor",
            "recall_without_factor",
            "failure_rate_without_factor",
            "risk_difference",
            "recall_difference",
            "factor_percentage",
            "failure_contribution_percentage",
        ],
        isolation,
    )

    print(
        f"[OK] {isolation_csv}"
    )

    # ------------------------------------------------------------------------
    # CSV 2
    # ------------------------------------------------------------------------

    csv_write(
        ranking_csv,
        [
            "rank",
            "factor",
            "objects_with_factor",
            "failures_with_factor",
            "tp_with_factor",
            "recall_with_factor",
            "failure_rate_with_factor",
            "objects_without_factor",
            "failures_without_factor",
            "tp_without_factor",
            "recall_without_factor",
            "failure_rate_without_factor",
            "risk_difference",
            "recall_difference",
            "factor_percentage",
            "failure_contribution_percentage",
            "isolation_priority",
        ],
        ranking,
    )

    print(
        f"[OK] {ranking_csv}"
    )

    # ------------------------------------------------------------------------
    # CSV 3
    # ------------------------------------------------------------------------

    csv_write(
        interactions_csv,
        [
            "rank",
            "factor_a",
            "factor_b",
            "objects",
            "failures",
            "tp",
            "recall",
            "failure_rate",
            "failure_contribution_percentage",
        ],
        interactions,
    )

    print(
        f"[OK] {interactions_csv}"
    )

    # ------------------------------------------------------------------------
    # CSV 4
    # ------------------------------------------------------------------------

    csv_write(
        pairs_csv,
        [
            "factor_a",
            "factor_b",
            "both_objects",
            "both_failures",
            "both_failure_rate",
            "a_only_objects",
            "a_only_failures",
            "a_only_failure_rate",
            "b_only_objects",
            "b_only_failures",
            "b_only_failure_rate",
            "neither_objects",
            "neither_failures",
            "neither_failure_rate",
        ],
        pairs,
    )

    print(
        f"[OK] {pairs_csv}"
    )

    # ------------------------------------------------------------------------
    # CSV 5
    # ------------------------------------------------------------------------

    csv_write(
        distribution_csv,
        [
            "factor",
            "objects",
            "failures",
            "tp",
            "object_percentage",
            "failure_percentage",
        ],
        distribution,
    )

    print(
        f"[OK] {distribution_csv}"
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print(
        "Generando summary..."
    )

    summary_path = write_summary(
        rows,
        isolation,
        ranking,
        interactions,
        pairs,
    )

    print(
        f"[OK] {summary_path}"
    )

    # ------------------------------------------------------------------------
    # Consola
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON SMALL FAILURE FACTOR ISOLATION V1"
    )
    print("=" * 72)
    print()

    print(
        f"Objetos analizados:       {len(rows):,}"
    )

    print(
        f"PERSON TP:                {total_tp:,}"
    )

    print(
        f"PERSON FN:                {total_failures:,}"
    )

    print(
        f"PERSON Recall:            "
        f"{total_tp / len(rows):.4f}"
    )

    print()

    print(
        "TOP FACTORES POR AISLAMIENTO"
    )

    print()

    for row in ranking:

        print(
            f"{row['rank']}. "
            f"{row['factor']:<25} "
            f"Failures="
            f"{row['failures_with_factor']:>6,} "
            f"FailureRate="
            f"{row['failure_rate_with_factor'] * 100:>6.2f}% "
            f"RiskDiff="
            f"{row['risk_difference'] * 100:>6.2f}pp "
            f"Priority="
            f"{row['isolation_priority']:>6.2f}"
        )

    print()

    print(
        "TOP INTERACCIONES"
    )

    print()

    for row in interactions[:15]:

        name = (
            f"{row['factor_a']} + "
            f"{row['factor_b']}"
        )

        print(
            f"{row['rank']:>2}. "
            f"{name:<55} "
            f"Failures="
            f"{row['failures']:>6,} "
            f"FailureRate="
            f"{row['failure_rate'] * 100:>6.2f}%"
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


if __name__ == "__main__":
    main()