from __future__ import annotations

import csv
from pathlib import Path
from itertools import combinations
from collections import defaultdict


# ============================================================================
# SAR YOLO26
# PERSON SMALL FAILURE CONSISTENCY ANALYSIS V1
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()

# ---------------------------------------------------------------------------
# IMPORTANTE:
#
# El script está ubicado en:
#
# baseline/
#   evaluation/
#     dataset_analysis/
#       detection_failure_analysis/
#         person/
#           small_failure_patterns/
#             analyze_person_small_failure_consistency_v1.py
#
# Por tanto:
#
# parents[0] = small_failure_patterns
# parents[1] = person
# parents[2] = detection_failure_analysis
# parents[3] = dataset_analysis
# parents[4] = evaluation
# parents[5] = baseline
# ---------------------------------------------------------------------------

BASE = SCRIPT_PATH.parents[5]

REPORTS_DIR = (
    BASE
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)

PATTERNS_REPORTS_DIR = (
    BASE
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_patterns_v1"
    / "reports"
)

INTERACTIONS_REPORTS_DIR = (
    BASE
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_interactions_v1"
    / "reports"
)


# ============================================================================
# INPUT REPORTS
# ============================================================================

INPUTS = {
    "patterns_objects": (
        PATTERNS_REPORTS_DIR
        / "person_small_failure_patterns_objects_v1.csv"
    ),

    "interactions_objects": (
        INTERACTIONS_REPORTS_DIR
        / "person_small_failure_interactions_objects_v1.csv"
    ),

    "factor_contribution": (
        REPORTS_DIR
        / "person_small_failure_factor_contribution_v1.csv"
    ),

    "factor_ranking": (
        REPORTS_DIR
        / "person_small_failure_factor_ranking_v1.csv"
    ),

    "factor_interactions": (
        REPORTS_DIR
        / "person_small_failure_factor_interactions_v1.csv"
    ),

    "priority": (
        REPORTS_DIR
        / "person_small_failure_priority_v1.csv"
    ),

    "priority_ranking": (
        REPORTS_DIR
        / "person_small_failure_priority_ranking_v1.csv"
    ),

    "priority_interactions": (
        REPORTS_DIR
        / "person_small_failure_priority_interactions_v1.csv"
    ),

    "priority_pairs": (
        REPORTS_DIR
        / "person_small_failure_priority_factor_pairs_v1.csv"
    ),

    "isolation": (
        REPORTS_DIR
        / "person_small_failure_factor_isolation_v1.csv"
    ),

    "isolation_ranking": (
        REPORTS_DIR
        / "person_small_failure_factor_isolation_ranking_v1.csv"
    ),

    "isolation_interactions": (
        REPORTS_DIR
        / "person_small_failure_factor_isolation_interactions_v1.csv"
    ),

    "isolation_pairs": (
        REPORTS_DIR
        / "person_small_failure_factor_isolation_factor_pairs_v1.csv"
    ),

    "isolation_distribution": (
        REPORTS_DIR
        / "person_small_failure_factor_isolation_distribution_v1.csv"
    ),
}


# ============================================================================
# OUTPUTS
# ============================================================================

OUTPUTS = {
    "factor_consistency": (
        REPORTS_DIR
        / "person_small_failure_factor_consistency_v1.csv"
    ),

    "interaction_consistency": (
        REPORTS_DIR
        / "person_small_failure_interaction_consistency_v1.csv"
    ),

    "factor_consistency_ranking": (
        REPORTS_DIR
        / "person_small_failure_factor_consistency_ranking_v1.csv"
    ),

    "consistency_matrix": (
        REPORTS_DIR
        / "person_small_failure_consistency_matrix_v1.csv"
    ),

    "consistency_summary": (
        REPORTS_DIR
        / "PERSON_SMALL_FAILURE_CONSISTENCY_V1_SUMMARY.txt"
    ),
}


# ============================================================================
# CONSTANTES
# ============================================================================

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


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path):
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

        rows = list(reader)
        fieldnames = reader.fieldnames or []

    return rows, fieldnames


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_header(title):
    print()
    print("=" * 72)
    print(f"# {title}")
    print("=" * 72)


# ============================================================================
# COMPROBACIÓN DE REPORTS
# ============================================================================

def check_reports():

    print("Comprobando reports...")

    missing = []

    for key, path in INPUTS.items():

        if path.exists() and path.stat().st_size > 0:
            print(f"[OK] {key}")
        else:
            print(f"[MISSING] {key}")
            missing.append((key, path))

    if missing:

        print()
        print("REPORTS FALTANTES:")
        print()

        for key, path in missing:
            print(key)
            print(path)
            print()

        raise FileNotFoundError(
            "Faltan uno o más reports necesarios para el análisis."
        )


# ============================================================================
# CARGA DE REPORTS
# ============================================================================

def load_report(key):

    rows, fields = read_csv(INPUTS[key])

    return rows, fields


# ============================================================================
# DETECCIÓN DE FACTORES DESDE interactions
# ============================================================================

def parse_interaction_factors(interaction):

    interaction = normalize(interaction)

    if not interaction:
        return []

    parts = [
        normalize(x)
        for x in interaction.split("+")
        if normalize(x)
    ]

    return parts


# ============================================================================
# CARGAR PATTERNS OBJECTS
# ============================================================================

def load_patterns_objects():

    rows, fields = load_report("patterns_objects")

    required = [
        "image",
        "person_gt_index",
        "area",
        "prediction_relation",
        "dominant_pattern",
    ]

    missing = [
        col
        for col in required
        if col not in fields
    ]

    if missing:
        raise ValueError(
            "Faltan columnas en patterns_objects:\n"
            + "\n".join(missing)
        )

    return rows


# ============================================================================
# CARGAR INTERACTIONS OBJECTS
# ============================================================================

def load_interactions_objects():

    rows, fields = load_report("interactions_objects")

    return rows


# ============================================================================
# CARGAR FACTOR CONTRIBUTION
# ============================================================================

def load_factor_contribution():

    rows, fields = load_report("factor_contribution")

    required = [
        "factor",
        "failures",
        "contribution_percentage",
        "interaction_count",
        "interaction_percentage",
    ]

    missing = [
        col
        for col in required
        if col not in fields
    ]

    if missing:
        raise ValueError(
            "Faltan columnas en factor_contribution:\n"
            + "\n".join(missing)
        )

    return rows


# ============================================================================
# CARGAR FACTOR RANKING
# ============================================================================

def load_factor_ranking():

    rows, fields = load_report("factor_ranking")

    required = [
        "rank",
        "factor",
        "failures",
        "contribution_percentage",
        "interaction_count",
        "interaction_percentage",
    ]

    missing = [
        col
        for col in required
        if col not in fields
    ]

    if missing:
        raise ValueError(
            "Faltan columnas en factor_ranking:\n"
            + "\n".join(missing)
        )

    return rows


# ============================================================================
# CARGAR FACTOR INTERACTIONS
# ============================================================================

def load_factor_interactions():

    rows, fields = load_report("factor_interactions")

    required = [
        "factor_a",
        "factor_b",
        "interaction",
        "failures",
        "percentage_of_total_failures",
        "occurrences",
    ]

    missing = [
        col
        for col in required
        if col not in fields
    ]

    if missing:
        raise ValueError(
            "Faltan columnas en factor_interactions:\n"
            + "\n".join(missing)
        )

    return rows


# ============================================================================
# CARGAR PRIORITY
# ============================================================================

def load_priority():

    rows, fields = load_report("priority")

    return rows


def load_priority_ranking():

    rows, fields = load_report("priority_ranking")

    return rows


def load_priority_interactions():

    rows, fields = load_report("priority_interactions")

    return rows


def load_priority_pairs():

    rows, fields = load_report("priority_pairs")

    return rows


# ============================================================================
# CARGAR ISOLATION
# ============================================================================

def load_isolation():

    rows, fields = load_report("isolation")

    return rows


def load_isolation_ranking():

    rows, fields = load_report("isolation_ranking")

    return rows


def load_isolation_interactions():

    rows, fields = load_report("isolation_interactions")

    return rows


def load_isolation_pairs():

    rows, fields = load_report("isolation_pairs")

    return rows


def load_isolation_distribution():

    rows, fields = load_report("isolation_distribution")

    return rows


# ============================================================================
# CONSISTENCIA DE FACTORES
# ============================================================================

def calculate_factor_consistency(
    contribution_rows,
    ranking_rows,
    priority_rows,
    isolation_rows,
):
    contribution = {
        normalize(row["factor"]): row
        for row in contribution_rows
    }

    ranking = {
        normalize(row["factor"]): row
        for row in ranking_rows
    }

    priority = {
        normalize(row["factor"]): row
        for row in priority_rows
    }

    isolation = {
        normalize(row["factor"]): row
        for row in isolation_rows
    }

    rows = []

    for factor in FACTORS:

        c = contribution.get(factor, {})
        r = ranking.get(factor, {})
        p = priority.get(factor, {})
        i = isolation.get(factor, {})

        contribution_value = safe_float(
            c.get("contribution_percentage")
        )

        ranking_value = safe_int(
            r.get("rank")
        )

        priority_value = safe_float(
            p.get("priority")
        )

        isolation_priority = safe_float(
            i.get("priority")
        )

        risk_diff = safe_float(
            i.get("risk_difference_pp")
        )

        failures = safe_int(
            c.get("failures")
        )

        rows.append({
            "factor": factor,
            "failures": failures,
            "contribution_percentage": round(
                contribution_value,
                6
            ),
            "original_rank": ranking_value,
            "priority_score": round(
                priority_value,
                6
            ),
            "isolation_priority": round(
                isolation_priority,
                6
            ),
            "risk_difference_pp": round(
                risk_diff,
                6
            ),
        })

    return rows


# ============================================================================
# RANKING DE CONSISTENCIA
# ============================================================================

def calculate_consistency_ranking(rows):

    result = []

    for row in rows:

        contribution = safe_float(
            row["contribution_percentage"]
        )

        priority = safe_float(
            row["priority_score"]
        )

        isolation = safe_float(
            row["isolation_priority"]
        )

        # -------------------------------------------------------------------
        # Consistencia:
        #
        # Un factor es consistente cuando aparece con fuerza:
        #
        # 1. En contribución
        # 2. En prioridad
        # 3. En aislamiento
        #
        # Se normaliza cada componente para evitar que una métrica domine
        # artificialmente el resultado.
        # -------------------------------------------------------------------

        contribution_norm = min(
            contribution / 100.0,
            1.0
        )

        priority_norm = min(
            priority / 100.0,
            1.0
        )

        isolation_norm = min(
            isolation / 100.0,
            1.0
        )

        consistency_score = (
            0.40 * contribution_norm
            + 0.35 * priority_norm
            + 0.25 * isolation_norm
        ) * 100.0

        result.append({
            **row,
            "consistency_score": round(
                consistency_score,
                6
            ),
        })

    result.sort(
        key=lambda x: x["consistency_score"],
        reverse=True
    )

    for idx, row in enumerate(result, start=1):
        row["consistency_rank"] = idx

    return result


# ============================================================================
# CONSISTENCIA DE INTERACCIONES
# ============================================================================

def calculate_interaction_consistency(
    factor_interactions,
    priority_interactions,
    isolation_interactions,
):
    priority_map = {}

    for row in priority_interactions:

        interaction = normalize(
            row.get("interaction")
        )

        priority_map[interaction] = row

    isolation_map = {}

    for row in isolation_interactions:

        interaction = normalize(
            row.get("interaction")
        )

        isolation_map[interaction] = row

    rows = []

    for row in factor_interactions:

        interaction = normalize(
            row.get("interaction")
        )

        failures = safe_int(
            row.get("failures")
        )

        percentage = safe_float(
            row.get("percentage_of_total_failures")
        )

        occurrences = safe_int(
            row.get("occurrences")
        )

        p = priority_map.get(
            interaction,
            {}
        )

        i = isolation_map.get(
            interaction,
            {}
        )

        priority_score = safe_float(
            p.get("priority")
        )

        isolation_priority = safe_float(
            i.get("priority")
        )

        # -------------------------------------------------------------------
        # Score de consistencia de interacción
        # -------------------------------------------------------------------

        percentage_norm = min(
            percentage / 100.0,
            1.0
        )

        priority_norm = min(
            priority_score / 100.0,
            1.0
        )

        isolation_norm = min(
            isolation_priority / 100.0,
            1.0
        )

        consistency_score = (
            0.40 * percentage_norm
            + 0.35 * priority_norm
            + 0.25 * isolation_norm
        ) * 100.0

        rows.append({
            "interaction": interaction,
            "failures": failures,
            "percentage_of_total_failures": round(
                percentage,
                6
            ),
            "occurrences": occurrences,
            "priority_score": round(
                priority_score,
                6
            ),
            "isolation_priority": round(
                isolation_priority,
                6
            ),
            "consistency_score": round(
                consistency_score,
                6
            ),
        })

    rows.sort(
        key=lambda x: x["consistency_score"],
        reverse=True
    )

    for idx, row in enumerate(rows, start=1):
        row["consistency_rank"] = idx

    return rows


# ============================================================================
# MATRIZ DE CONSISTENCIA ENTRE FACTORES
# ============================================================================

def calculate_consistency_matrix(
    factor_rows,
    interaction_rows,
):

    factor_map = {
        row["factor"]: row
        for row in factor_rows
    }

    interaction_map = {
        row["interaction"]: row
        for row in interaction_rows
    }

    result = []

    for factor_a, factor_b in combinations(
        FACTORS,
        2
    ):

        interaction = (
            f"{factor_a} + {factor_b}"
        )

        reverse_interaction = (
            f"{factor_b} + {factor_a}"
        )

        row = interaction_map.get(
            interaction
        )

        if row is None:
            row = interaction_map.get(
                reverse_interaction,
                {}
            )

        failures = safe_int(
            row.get("failures")
        )

        percentage = safe_float(
            row.get(
                "percentage_of_total_failures"
            )
        )

        consistency = safe_float(
            row.get(
                "consistency_score"
            )
        )

        factor_a_score = safe_float(
            factor_map[factor_a].get(
                "consistency_score"
            )
        )

        factor_b_score = safe_float(
            factor_map[factor_b].get(
                "consistency_score"
            )
        )

        combined_score = (
            0.40 * consistency
            + 0.30 * factor_a_score
            + 0.30 * factor_b_score
        )

        result.append({
            "factor_a": factor_a,
            "factor_b": factor_b,
            "interaction": interaction,
            "failures": failures,
            "percentage_of_total_failures": round(
                percentage,
                6
            ),
            "interaction_consistency": round(
                consistency,
                6
            ),
            "factor_a_consistency": round(
                factor_a_score,
                6
            ),
            "factor_b_consistency": round(
                factor_b_score,
                6
            ),
            "combined_consistency_score": round(
                combined_score,
                6
            ),
        })

    result.sort(
        key=lambda x: x["combined_consistency_score"],
        reverse=True
    )

    for idx, row in enumerate(
        result,
        start=1
    ):
        row["rank"] = idx

    return result


# ============================================================================
# SUMMARY
# ============================================================================

def generate_summary(
    factor_rows,
    interaction_rows,
    matrix_rows,
    patterns_objects,
    interactions_objects,
):

    path = OUTPUTS["consistency_summary"]

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_objects = len(
        patterns_objects
    )

    total_interaction_objects = len(
        interactions_objects
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "========================================================================\n"
        )
        f.write(
            "SAR YOLO26 - PERSON SMALL FAILURE CONSISTENCY ANALYSIS V1\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        f.write(
            f"Objects PATTERNS: {total_objects}\n"
        )

        f.write(
            f"Objects INTERACTIONS: {total_interaction_objects}\n\n"
        )

        f.write(
            f"Factores analizados: {len(factor_rows)}\n"
        )

        f.write(
            f"Interacciones analizadas: {len(interaction_rows)}\n"
        )

        f.write(
            f"Pares de factores: {len(matrix_rows)}\n\n"
        )

        f.write(
            "TOP FACTORES POR CONSISTENCIA\n\n"
        )

        for idx, row in enumerate(
            factor_rows[:7],
            start=1
        ):

            f.write(
                f"{idx}. "
                f"{row['factor']:<25} "
                f"Failures={row['failures']:>6} "
                f"Contribution={row['contribution_percentage']:>7.2f}% "
                f"Priority={row['priority_score']:>7.2f} "
                f"Isolation={row['isolation_priority']:>7.2f} "
                f"Consistency={row['consistency_score']:>7.2f}\n"
            )

        f.write(
            "\nTOP INTERACCIONES POR CONSISTENCIA\n\n"
        )

        for idx, row in enumerate(
            interaction_rows[:15],
            start=1
        ):

            f.write(
                f"{idx:>2}. "
                f"{row['interaction']:<60} "
                f"Failures={row['failures']:>6} "
                f"Percentage={row['percentage_of_total_failures']:>7.2f}% "
                f"Consistency={row['consistency_score']:>7.2f}\n"
            )

        f.write(
            "\nTOP PARES DE FACTORES\n\n"
        )

        for idx, row in enumerate(
            matrix_rows[:15],
            start=1
        ):

            f.write(
                f"{idx:>2}. "
                f"{row['interaction']:<60} "
                f"Failures={row['failures']:>6} "
                f"Score={row['combined_consistency_score']:>7.2f}\n"
            )

        f.write(
            "\n========================================================================\n"
        )
        f.write(
            "FIN DEL ANALISIS\n"
        )
        f.write(
            "========================================================================\n"
        )

    return path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "SAR YOLO26 - PERSON SMALL FAILURE CONSISTENCY ANALYSIS V1"
    )

    print()
    print("SCRIPT:")
    print(SCRIPT_PATH)

    print()
    print("BASE:")
    print(BASE)

    print()
    print("Reports:")
    print(REPORTS_DIR)

    # -----------------------------------------------------------------------
    # Comprobar que estamos realmente en baseline
    # -----------------------------------------------------------------------

    if BASE.name.lower() != "baseline":

        raise RuntimeError(
            "La raíz BASE calculada no parece ser 'baseline'.\n"
            f"BASE detectada: {BASE}\n"
            f"Script: {SCRIPT_PATH}"
        )

    # -----------------------------------------------------------------------
    # Crear reports si no existe
    # -----------------------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------------------------
    # Comprobar inputs
    # -----------------------------------------------------------------------

    print()

    check_reports()

    # -----------------------------------------------------------------------
    # Cargar PATTERNS
    # -----------------------------------------------------------------------

    print()
    print("Cargando PATTERNS objects...")

    patterns_objects = load_patterns_objects()

    print(
        f"[OK] PATTERNS objects: "
        f"{len(patterns_objects):,}"
    )

    # -----------------------------------------------------------------------
    # Cargar INTERACTIONS
    # -----------------------------------------------------------------------

    print()
    print("Cargando INTERACTIONS objects...")

    interactions_objects = (
        load_interactions_objects()
    )

    print(
        f"[OK] INTERACTIONS objects: "
        f"{len(interactions_objects):,}"
    )

    # -----------------------------------------------------------------------
    # Cargar contribución
    # -----------------------------------------------------------------------

    print()
    print("Cargando contribución de factores...")

    contribution = (
        load_factor_contribution()
    )

    print(
        f"[OK] Factores: "
        f"{len(contribution)}"
    )

    # -----------------------------------------------------------------------
    # Cargar ranking
    # -----------------------------------------------------------------------

    print()
    print("Cargando ranking original...")

    ranking = load_factor_ranking()

    print(
        f"[OK] Ranking: "
        f"{len(ranking)}"
    )

    # -----------------------------------------------------------------------
    # Cargar interacciones
    # -----------------------------------------------------------------------

    print()
    print("Cargando interacciones...")

    factor_interactions = (
        load_factor_interactions()
    )

    print(
        f"[OK] Interacciones: "
        f"{len(factor_interactions)}"
    )

    # -----------------------------------------------------------------------
    # Cargar priority
    # -----------------------------------------------------------------------

    print()
    print("Cargando prioridad...")

    priority = load_priority()

    print(
        f"[OK] Priority: "
        f"{len(priority)}"
    )

    priority_ranking = (
        load_priority_ranking()
    )

    priority_interactions = (
        load_priority_interactions()
    )

    priority_pairs = (
        load_priority_pairs()
    )

    print(
        f"[OK] Priority ranking: "
        f"{len(priority_ranking)}"
    )

    print(
        f"[OK] Priority interactions: "
        f"{len(priority_interactions)}"
    )

    print(
        f"[OK] Priority pairs: "
        f"{len(priority_pairs)}"
    )

    # -----------------------------------------------------------------------
    # Cargar isolation
    # -----------------------------------------------------------------------

    print()
    print("Cargando aislamiento...")

    isolation = load_isolation()

    isolation_ranking = (
        load_isolation_ranking()
    )

    isolation_interactions = (
        load_isolation_interactions()
    )

    isolation_pairs = (
        load_isolation_pairs()
    )

    isolation_distribution = (
        load_isolation_distribution()
    )

    print(
        f"[OK] Isolation: "
        f"{len(isolation)}"
    )

    print(
        f"[OK] Isolation ranking: "
        f"{len(isolation_ranking)}"
    )

    print(
        f"[OK] Isolation interactions: "
        f"{len(isolation_interactions)}"
    )

    print(
        f"[OK] Isolation pairs: "
        f"{len(isolation_pairs)}"
    )

    print(
        f"[OK] Isolation distribution: "
        f"{len(isolation_distribution)}"
    )

    # -----------------------------------------------------------------------
    # CONSISTENCIA DE FACTORES
    # -----------------------------------------------------------------------

    print()
    print("Calculando consistencia de factores...")

    factor_consistency = (
        calculate_factor_consistency(
            contribution,
            ranking,
            priority,
            isolation,
        )
    )

    factor_consistency_ranking = (
        calculate_consistency_ranking(
            factor_consistency
        )
    )

    print(
        f"[OK] Factores consistentes: "
        f"{len(factor_consistency_ranking)}"
    )

    # -----------------------------------------------------------------------
    # CONSISTENCIA DE INTERACCIONES
    # -----------------------------------------------------------------------

    print()
    print("Calculando consistencia de interacciones...")

    interaction_consistency = (
        calculate_interaction_consistency(
            factor_interactions,
            priority_interactions,
            isolation_interactions,
        )
    )

    print(
        f"[OK] Interacciones consistentes: "
        f"{len(interaction_consistency)}"
    )

    # -----------------------------------------------------------------------
    # MATRIZ
    # -----------------------------------------------------------------------

    print()
    print("Calculando matriz de consistencia...")

    consistency_matrix = (
        calculate_consistency_matrix(
            factor_consistency_ranking,
            interaction_consistency,
        )
    )

    print(
        f"[OK] Pares calculados: "
        f"{len(consistency_matrix)}"
    )

    # -----------------------------------------------------------------------
    # GENERAR CSV FACTORES
    # -----------------------------------------------------------------------

    print()
    print("Generando reports...")

    write_csv(
        OUTPUTS["factor_consistency"],
        [
            "factor",
            "failures",
            "contribution_percentage",
            "original_rank",
            "priority_score",
            "isolation_priority",
            "risk_difference_pp",
        ],
        factor_consistency,
    )

    print(
        f"[OK] {OUTPUTS['factor_consistency']}"
    )

    # -----------------------------------------------------------------------
    # RANKING
    # -----------------------------------------------------------------------

    write_csv(
        OUTPUTS["factor_consistency_ranking"],
        [
            "consistency_rank",
            "factor",
            "failures",
            "contribution_percentage",
            "original_rank",
            "priority_score",
            "isolation_priority",
            "risk_difference_pp",
            "consistency_score",
        ],
        factor_consistency_ranking,
    )

    print(
        f"[OK] {OUTPUTS['factor_consistency_ranking']}"
    )

    # -----------------------------------------------------------------------
    # INTERACCIONES
    # -----------------------------------------------------------------------

    write_csv(
        OUTPUTS["interaction_consistency"],
        [
            "consistency_rank",
            "interaction",
            "failures",
            "percentage_of_total_failures",
            "occurrences",
            "priority_score",
            "isolation_priority",
            "consistency_score",
        ],
        interaction_consistency,
    )

    print(
        f"[OK] {OUTPUTS['interaction_consistency']}"
    )

    # -----------------------------------------------------------------------
    # MATRIZ
    # -----------------------------------------------------------------------

    write_csv(
        OUTPUTS["consistency_matrix"],
        [
            "rank",
            "factor_a",
            "factor_b",
            "interaction",
            "failures",
            "percentage_of_total_failures",
            "interaction_consistency",
            "factor_a_consistency",
            "factor_b_consistency",
            "combined_consistency_score",
        ],
        consistency_matrix,
    )

    print(
        f"[OK] {OUTPUTS['consistency_matrix']}"
    )

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    print()
    print("Generando summary...")

    summary_path = generate_summary(
        factor_consistency_ranking,
        interaction_consistency,
        consistency_matrix,
        patterns_objects,
        interactions_objects,
    )

    print(
        f"[OK] {summary_path}"
    )

    # -----------------------------------------------------------------------
    # RESULTADO
    # -----------------------------------------------------------------------

    print_header(
        "RESULTADO PERSON SMALL FAILURE CONSISTENCY V1"
    )

    print()
    print(
        f"Objetos PATTERNS:       "
        f"{len(patterns_objects):,}"
    )

    print(
        f"Objetos INTERACTIONS:   "
        f"{len(interactions_objects):,}"
    )

    print(
        f"Factores analizados:    "
        f"{len(factor_consistency_ranking)}"
    )

    print(
        f"Interacciones:          "
        f"{len(interaction_consistency)}"
    )

    print(
        f"Pares de factores:      "
        f"{len(consistency_matrix)}"
    )

    print()
    print("TOP FACTORES POR CONSISTENCIA")
    print()

    for row in factor_consistency_ranking[:7]:

        print(
            f"{row['consistency_rank']}. "
            f"{row['factor']:<25} "
            f"Failures={row['failures']:>6} "
            f"Contribution={row['contribution_percentage']:>7.2f}% "
            f"Priority={row['priority_score']:>7.2f} "
            f"Isolation={row['isolation_priority']:>7.2f} "
            f"Consistency={row['consistency_score']:>7.2f}"
        )

    print()
    print("TOP INTERACCIONES POR CONSISTENCIA")
    print()

    for row in interaction_consistency[:15]:

        print(
            f"{row['consistency_rank']:>2}. "
            f"{row['interaction']:<60} "
            f"Failures={row['failures']:>6} "
            f"Percentage={row['percentage_of_total_failures']:>7.2f}% "
            f"Consistency={row['consistency_score']:>7.2f}"
        )

    print()
    print("[OK] Reports generados.")
    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )


if __name__ == "__main__":
    main()