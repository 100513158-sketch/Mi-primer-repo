from pathlib import Path
import csv
from itertools import combinations


# ============================================================================
# SAR YOLO26
# PERSON SMALL FAILURE PRIORITY ANALYSIS V1
# ============================================================================


SCRIPT_PATH = Path(__file__).resolve()


# ============================================================================
# LOCALIZAR BASELINE DE FORMA ROBUSTA
# ============================================================================

def find_baseline_dir(script_path: Path) -> Path:
    """
    Busca el directorio 'baseline' entre los padres del script.

    No se utiliza parents[n] para evitar errores si cambia
    la profundidad de la estructura de directorios.
    """

    for parent in script_path.parents:
        if parent.name.lower() == "baseline":
            return parent

    raise RuntimeError(
        "No se pudo localizar el directorio 'baseline'.\n"
        f"Script: {script_path}"
    )


BASELINE_DIR = find_baseline_dir(SCRIPT_PATH)


# ============================================================================
# DIRECTORIOS
# ============================================================================

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)


# ============================================================================
# INPUTS
# ============================================================================

INPUT_CONTRIBUTION = (
    REPORTS_DIR
    / "person_small_failure_factor_contribution_v1.csv"
)

INPUT_RANKING = (
    REPORTS_DIR
    / "person_small_failure_factor_ranking_v1.csv"
)

INPUT_INTERACTIONS = (
    REPORTS_DIR
    / "person_small_failure_factor_interactions_v1.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

OUTPUT_PRIORITY = (
    REPORTS_DIR
    / "person_small_failure_priority_v1.csv"
)

OUTPUT_PRIORITY_RANKING = (
    REPORTS_DIR
    / "person_small_failure_priority_ranking_v1.csv"
)

OUTPUT_PRIORITY_INTERACTIONS = (
    REPORTS_DIR
    / "person_small_failure_priority_interactions_v1.csv"
)

OUTPUT_PRIORITY_FACTOR_PAIRS = (
    REPORTS_DIR
    / "person_small_failure_priority_factor_pairs_v1.csv"
)

OUTPUT_SUMMARY = (
    REPORTS_DIR
    / "PERSON_SMALL_FAILURE_PRIORITY_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

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


def ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    """
    Lee un CSV y devuelve:
        rows, fieldnames
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []

        rows = list(reader)

    return rows, fieldnames


def write_csv(path: Path, fieldnames, rows):
    """
    Escribe un CSV UTF-8.
    """

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


# ============================================================================
# CARGAR CONTRIBUCIÓN
# ============================================================================

def load_factor_contribution():

    rows, fieldnames = read_csv(
        INPUT_CONTRIBUTION
    )

    required = {
        "factor",
        "failures",
        "contribution_percentage",
        "interaction_count",
        "interaction_percentage",
    }

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            "Faltan columnas en "
            f"{INPUT_CONTRIBUTION}:\n"
            f"{sorted(missing)}"
        )

    factors = []

    for row in rows:

        factors.append(
            {
                "factor": row["factor"],
                "failures": safe_int(
                    row["failures"]
                ),
                "contribution_percentage":
                    safe_float(
                        row["contribution_percentage"]
                    ),
                "interaction_count":
                    safe_int(
                        row["interaction_count"]
                    ),
                "interaction_percentage":
                    safe_float(
                        row["interaction_percentage"]
                    ),
            }
        )

    return factors


# ============================================================================
# CARGAR RANKING ORIGINAL
# ============================================================================

def load_factor_ranking():

    rows, fieldnames = read_csv(
        INPUT_RANKING
    )

    required = {
        "rank",
        "factor",
        "failures",
        "contribution_percentage",
        "interaction_count",
        "interaction_percentage",
    }

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            "Faltan columnas en "
            f"{INPUT_RANKING}:\n"
            f"{sorted(missing)}"
        )

    ranking = []

    for row in rows:

        ranking.append(
            {
                "rank": safe_int(
                    row["rank"]
                ),
                "factor": row["factor"],
                "failures": safe_int(
                    row["failures"]
                ),
                "contribution_percentage":
                    safe_float(
                        row["contribution_percentage"]
                    ),
                "interaction_count":
                    safe_int(
                        row["interaction_count"]
                    ),
                "interaction_percentage":
                    safe_float(
                        row["interaction_percentage"]
                    ),
            }
        )

    ranking.sort(
        key=lambda x: x["rank"]
    )

    return ranking


# ============================================================================
# CARGAR INTERACCIONES
# ============================================================================

def load_factor_interactions():

    rows, fieldnames = read_csv(
        INPUT_INTERACTIONS
    )

    required = {
        "factor_a",
        "factor_b",
        "interaction",
        "failures",
        "percentage_of_total_failures",
        "occurrences",
    }

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            "Faltan columnas en "
            f"{INPUT_INTERACTIONS}:\n"
            f"{sorted(missing)}"
        )

    interactions = []

    for row in rows:

        interactions.append(
            {
                "factor_a": row["factor_a"],
                "factor_b": row["factor_b"],
                "interaction": row["interaction"],
                "failures": safe_int(
                    row["failures"]
                ),
                "percentage_of_total_failures":
                    safe_float(
                        row[
                            "percentage_of_total_failures"
                        ]
                    ),
                "occurrences": safe_int(
                    row["occurrences"]
                ),
            }
        )

    return interactions


# ============================================================================
# NORMALIZAR FACTOR
# ============================================================================

def normalize_factor_name(value):

    return (
        str(value)
        .strip()
        .upper()
    )


# ============================================================================
# CALCULAR SCORE DE PRIORIDAD
# ============================================================================

def calculate_priority_score(
    contribution_percentage,
    interaction_percentage,
    interaction_count,
):
    """
    Score compuesto de prioridad.

    Componentes:

    - 60% contribución individual
    - 25% interacción
    - 15% frecuencia de aparición en interacciones

    La frecuencia se normaliza posteriormente.
    """

    return (
        0.60 * contribution_percentage
        + 0.25 * interaction_percentage
        + 0.15 * interaction_count
    )


# ============================================================================
# CALCULAR PRIORIDAD DE FACTORES
# ============================================================================

def calculate_factor_priority(
    factors,
    interactions,
):

    if not factors:
        return []

    max_interaction_count = max(
        (
            factor["interaction_count"]
            for factor in factors
        ),
        default=1,
    )

    if max_interaction_count <= 0:
        max_interaction_count = 1

    priority_rows = []

    for factor in factors:

        name = normalize_factor_name(
            factor["factor"]
        )

        normalized_interaction_count = (
            factor["interaction_count"]
            / max_interaction_count
        ) * 100.0

        score = calculate_priority_score(
            factor["contribution_percentage"],
            factor["interaction_percentage"],
            normalized_interaction_count,
        )

        priority_rows.append(
            {
                "factor": name,
                "failures":
                    factor["failures"],
                "contribution_percentage":
                    factor[
                        "contribution_percentage"
                    ],
                "interaction_count":
                    factor[
                        "interaction_count"
                    ],
                "interaction_percentage":
                    factor[
                        "interaction_percentage"
                    ],
                "normalized_interaction_count":
                    normalized_interaction_count,
                "priority_score":
                    score,
            }
        )

    priority_rows.sort(
        key=lambda x: (
            x["priority_score"],
            x["failures"],
        ),
        reverse=True,
    )

    for index, row in enumerate(
        priority_rows,
        start=1
    ):
        row["priority_rank"] = index

    return priority_rows


# ============================================================================
# CALCULAR PRIORIDAD DE INTERACCIONES
# ============================================================================

def calculate_interaction_priority(
    interactions,
):

    if not interactions:
        return []

    max_failures = max(
        (
            item["failures"]
            for item in interactions
        ),
        default=1,
    )

    if max_failures <= 0:
        max_failures = 1

    result = []

    for item in interactions:

        normalized_failures = (
            item["failures"]
            / max_failures
        ) * 100.0

        score = (
            0.70
            * item[
                "percentage_of_total_failures"
            ]
            + 0.30
            * normalized_failures
        )

        result.append(
            {
                "factor_a":
                    item["factor_a"],
                "factor_b":
                    item["factor_b"],
                "interaction":
                    item["interaction"],
                "failures":
                    item["failures"],
                "percentage_of_total_failures":
                    item[
                        "percentage_of_total_failures"
                    ],
                "occurrences":
                    item["occurrences"],
                "normalized_failures":
                    normalized_failures,
                "priority_score":
                    score,
            }
        )

    result.sort(
        key=lambda x: (
            x["priority_score"],
            x["failures"],
        ),
        reverse=True,
    )

    for index, row in enumerate(
        result,
        start=1
    ):
        row["priority_rank"] = index

    return result


# ============================================================================
# GENERAR PARES DE FACTORES
# ============================================================================

def calculate_factor_pairs(
    factor_priority,
    interaction_priority,
):

    factor_map = {
        row["factor"]: row
        for row in factor_priority
    }

    rows = []

    for item in interaction_priority:

        factor_a = normalize_factor_name(
            item["factor_a"]
        )

        factor_b = normalize_factor_name(
            item["factor_b"]
        )

        a = factor_map.get(
            factor_a
        )

        b = factor_map.get(
            factor_b
        )

        if a is None or b is None:
            continue

        mean_factor_priority = (
            a["priority_score"]
            + b["priority_score"]
        ) / 2.0

        combined_score = (
            0.50
            * item["priority_score"]
            + 0.50
            * mean_factor_priority
        )

        rows.append(
            {
                "factor_a":
                    factor_a,
                "factor_b":
                    factor_b,
                "interaction":
                    item["interaction"],
                "interaction_failures":
                    item["failures"],
                "interaction_percentage":
                    item[
                        "percentage_of_total_failures"
                    ],
                "factor_a_priority":
                    a["priority_score"],
                "factor_b_priority":
                    b["priority_score"],
                "mean_factor_priority":
                    mean_factor_priority,
                "combined_priority_score":
                    combined_score,
            }
        )

    rows.sort(
        key=lambda x: (
            x["combined_priority_score"],
            x["interaction_failures"],
        ),
        reverse=True,
    )

    for index, row in enumerate(
        rows,
        start=1
    ):
        row["priority_rank"] = index

    return rows


# ============================================================================
# CALCULAR COBERTURA
# ============================================================================

def calculate_coverage(
    factor_priority,
):

    total_contribution = sum(
        row["contribution_percentage"]
        for row in factor_priority
    )

    return total_contribution


# ============================================================================
# GENERAR SUMMARY
# ============================================================================

def generate_summary(
    factors,
    factor_priority,
    interaction_priority,
    factor_pairs,
):

    lines = []

    lines.append(
        "========================================================================"
    )

    lines.append(
        "# SAR YOLO26 - PERSON SMALL FAILURE PRIORITY ANALYSIS V1"
    )

    lines.append(
        "========================================================================"
    )

    lines.append("")

    lines.append(
        f"Factores analizados: {len(factors)}"
    )

    lines.append(
        f"Interacciones analizadas: "
        f"{len(interaction_priority)}"
    )

    lines.append(
        f"Pares de factores: "
        f"{len(factor_pairs)}"
    )

    lines.append("")

    # ------------------------------------------------------------------------
    # FACTORES
    # ------------------------------------------------------------------------

    lines.append(
        "## PRIORIDAD DE FACTORES"
    )

    lines.append("")

    for row in factor_priority:

        lines.append(
            f"{row['priority_rank']:>2}. "
            f"{row['factor']:<25} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Contribution="
            f"{row['contribution_percentage']:>7.2f}% "
            f"Interactions="
            f"{row['interaction_count']:>3} "
            f"Priority="
            f"{row['priority_score']:>7.2f}"
        )

    lines.append("")

    # ------------------------------------------------------------------------
    # INTERACCIONES
    # ------------------------------------------------------------------------

    lines.append(
        "## PRIORIDAD DE INTERACCIONES"
    )

    lines.append("")

    for row in interaction_priority[:20]:

        lines.append(
            f"{row['priority_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Percentage="
            f"{row['percentage_of_total_failures']:>7.2f}% "
            f"Priority="
            f"{row['priority_score']:>7.2f}"
        )

    lines.append("")

    # ------------------------------------------------------------------------
    # PARES
    # ------------------------------------------------------------------------

    lines.append(
        "## TOP FACTOR PAIRS"
    )

    lines.append("")

    for row in factor_pairs[:20]:

        lines.append(
            f"{row['priority_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures="
            f"{row['interaction_failures']:>6} "
            f"CombinedPriority="
            f"{row['combined_priority_score']:>7.2f}"
        )

    lines.append("")

    # ------------------------------------------------------------------------
    # INTERPRETACIÓN
    # ------------------------------------------------------------------------

    if factor_priority:

        top_factor = factor_priority[0]

        lines.append(
            "## PRINCIPAL FACTOR PRIORITARIO"
        )

        lines.append("")

        lines.append(
            f"{top_factor['factor']} "
            f"presenta la mayor prioridad compuesta "
            f"con un score de "
            f"{top_factor['priority_score']:.2f}."
        )

        lines.append(
            f"Su contribución individual es "
            f"{top_factor['contribution_percentage']:.2f}% "
            f"sobre los small failures representados."
        )

        lines.append("")

    if interaction_priority:

        top_interaction = (
            interaction_priority[0]
        )

        lines.append(
            "## PRINCIPAL INTERACCIÓN PRIORITARIA"
        )

        lines.append("")

        lines.append(
            f"{top_interaction['interaction']} "
            f"es la interacción prioritaria "
            f"con {top_interaction['failures']} "
            f"fallos y "
            f"{top_interaction['percentage_of_total_failures']:.2f}% "
            f"del total representado."
        )

        lines.append("")

    lines.append(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    lines.append("")

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print(
        "========================================================================"
    )
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE PRIORITY ANALYSIS V1"
    )
    print(
        "========================================================================"
    )
    print()

    print("Input reports:")

    print(
        f"  {INPUT_CONTRIBUTION}"
    )

    print(
        f"  {INPUT_RANKING}"
    )

    print(
        f"  {INPUT_INTERACTIONS}"
    )

    print()

    print("Output:")

    print(
        f"  {REPORTS_DIR}"
    )

    print()

    # ------------------------------------------------------------------------
    # PREPARAR
    # ------------------------------------------------------------------------

    ensure_reports_dir()

    # ------------------------------------------------------------------------
    # COMPROBAR INPUTS
    # ------------------------------------------------------------------------

    print(
        "Comprobando reports de factores..."
    )

    for path in [
        INPUT_CONTRIBUTION,
        INPUT_RANKING,
        INPUT_INTERACTIONS,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

        print(
            f"[OK] {path.name}"
        )

    print()

    # ------------------------------------------------------------------------
    # CARGAR
    # ------------------------------------------------------------------------

    print(
        "Cargando contribución de factores..."
    )

    factors = load_factor_contribution()

    print(
        f"[OK] Factores cargados: "
        f"{len(factors)}"
    )

    print()

    print(
        "Cargando ranking original..."
    )

    ranking = load_factor_ranking()

    print(
        f"[OK] Ranking cargado: "
        f"{len(ranking)}"
    )

    print()

    print(
        "Cargando interacciones..."
    )

    interactions = load_factor_interactions()

    print(
        f"[OK] Interacciones cargadas: "
        f"{len(interactions)}"
    )

    print()

    # ------------------------------------------------------------------------
    # FACTOR PRIORITY
    # ------------------------------------------------------------------------

    print(
        "Calculando prioridad de factores..."
    )

    factor_priority = calculate_factor_priority(
        factors,
        interactions,
    )

    print(
        f"[OK] Factores priorizados: "
        f"{len(factor_priority)}"
    )

    print()

    # ------------------------------------------------------------------------
    # INTERACTION PRIORITY
    # ------------------------------------------------------------------------

    print(
        "Calculando prioridad de interacciones..."
    )

    interaction_priority = (
        calculate_interaction_priority(
            interactions
        )
    )

    print(
        f"[OK] Interacciones priorizadas: "
        f"{len(interaction_priority)}"
    )

    print()

    # ------------------------------------------------------------------------
    # FACTOR PAIRS
    # ------------------------------------------------------------------------

    print(
        "Calculando pares de factores..."
    )

    factor_pairs = calculate_factor_pairs(
        factor_priority,
        interaction_priority,
    )

    print(
        f"[OK] Pares calculados: "
        f"{len(factor_pairs)}"
    )

    print()

    # ------------------------------------------------------------------------
    # WRITE FACTOR PRIORITY
    # ------------------------------------------------------------------------

    print(
        "Generando report de prioridad..."
    )

    write_csv(
        OUTPUT_PRIORITY,
        [
            "priority_rank",
            "factor",
            "failures",
            "contribution_percentage",
            "interaction_count",
            "interaction_percentage",
            "normalized_interaction_count",
            "priority_score",
        ],
        factor_priority,
    )

    print(
        f"[OK] {OUTPUT_PRIORITY}"
    )

    # ------------------------------------------------------------------------
    # WRITE PRIORITY RANKING
    # ------------------------------------------------------------------------

    ranking_rows = []

    factor_priority_map = {
        row["factor"]: row
        for row in factor_priority
    }

    for original in ranking:

        factor = normalize_factor_name(
            original["factor"]
        )

        priority = factor_priority_map.get(
            factor
        )

        if priority is None:
            continue

        ranking_rows.append(
            {
                "original_rank":
                    original["rank"],
                "priority_rank":
                    priority["priority_rank"],
                "factor":
                    factor,
                "failures":
                    priority["failures"],
                "contribution_percentage":
                    priority[
                        "contribution_percentage"
                    ],
                "interaction_count":
                    priority[
                        "interaction_count"
                    ],
                "interaction_percentage":
                    priority[
                        "interaction_percentage"
                    ],
                "priority_score":
                    priority[
                        "priority_score"
                    ],
            }
        )

    write_csv(
        OUTPUT_PRIORITY_RANKING,
        [
            "original_rank",
            "priority_rank",
            "factor",
            "failures",
            "contribution_percentage",
            "interaction_count",
            "interaction_percentage",
            "priority_score",
        ],
        ranking_rows,
    )

    print(
        f"[OK] {OUTPUT_PRIORITY_RANKING}"
    )

    # ------------------------------------------------------------------------
    # WRITE INTERACTION PRIORITY
    # ------------------------------------------------------------------------

    write_csv(
        OUTPUT_PRIORITY_INTERACTIONS,
        [
            "priority_rank",
            "factor_a",
            "factor_b",
            "interaction",
            "failures",
            "percentage_of_total_failures",
            "occurrences",
            "normalized_failures",
            "priority_score",
        ],
        interaction_priority,
    )

    print(
        f"[OK] {OUTPUT_PRIORITY_INTERACTIONS}"
    )

    # ------------------------------------------------------------------------
    # WRITE FACTOR PAIRS
    # ------------------------------------------------------------------------

    write_csv(
        OUTPUT_PRIORITY_FACTOR_PAIRS,
        [
            "priority_rank",
            "factor_a",
            "factor_b",
            "interaction",
            "interaction_failures",
            "interaction_percentage",
            "factor_a_priority",
            "factor_b_priority",
            "mean_factor_priority",
            "combined_priority_score",
        ],
        factor_pairs,
    )

    print(
        f"[OK] {OUTPUT_PRIORITY_FACTOR_PAIRS}"
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    print()

    print(
        "Generando summary..."
    )

    generate_summary(
        factors,
        factor_priority,
        interaction_priority,
        factor_pairs,
    )

    print(
        f"[OK] {OUTPUT_SUMMARY}"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    print()

    print(
        "========================================================================"
    )

    print(
        "# RESULTADO PERSON SMALL FAILURE PRIORITY V1"
    )

    print(
        "========================================================================"
    )

    print()

    print(
        f"Factores analizados:      "
        f"{len(factors)}"
    )

    print(
        f"Interacciones analizadas:"
        f" {len(interaction_priority)}"
    )

    print(
        f"Pares de factores:        "
        f"{len(factor_pairs)}"
    )

    print()

    print(
        "TOP FACTORES PRIORITARIOS"
    )

    print()

    for row in factor_priority:

        print(
            f"{row['priority_rank']:>2}. "
            f"{row['factor']:<25} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Contribution="
            f"{row['contribution_percentage']:>7.2f}% "
            f"Priority="
            f"{row['priority_score']:>7.2f}"
        )

    print()

    print(
        "TOP INTERACCIONES PRIORITARIAS"
    )

    print()

    for row in interaction_priority[:15]:

        print(
            f"{row['priority_rank']:>2}. "
            f"{row['interaction']:<55} "
            f"Failures="
            f"{row['failures']:>6} "
            f"Percentage="
            f"{row['percentage_of_total_failures']:>7.2f}% "
            f"Priority="
            f"{row['priority_score']:>7.2f}"
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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()