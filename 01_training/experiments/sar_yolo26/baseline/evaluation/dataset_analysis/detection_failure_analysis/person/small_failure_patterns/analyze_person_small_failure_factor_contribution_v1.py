from pathlib import Path
import csv
from collections import defaultdict
from itertools import combinations


# ============================================================================
# SAR YOLO26 - PERSON SMALL FAILURE FACTOR CONTRIBUTION V1
# ============================================================================

SCRIPT_VERSION = "V1"


# ============================================================================
# PATHS
# ============================================================================

BASELINE_DIR = (
    Path(r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline")
)

INPUT_CSV = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_interactions_v1"
    / "reports"
    / "person_small_failure_interactions_v1.csv"
)

OUTPUT_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "reports"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_CONTRIBUTION = (
    OUTPUT_DIR
    / "person_small_failure_factor_contribution_v1.csv"
)

OUTPUT_RANKING = (
    OUTPUT_DIR
    / "person_small_failure_factor_ranking_v1.csv"
)

OUTPUT_INTERACTIONS = (
    OUTPUT_DIR
    / "person_small_failure_factor_interactions_v1.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "PERSON_SMALL_FAILURE_FACTOR_CONTRIBUTION_V1_SUMMARY.txt"
)


# ============================================================================
# CONSTANTS
# ============================================================================

SEPARATOR = " + "


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_factor(value):
    """
    Normaliza el nombre de un factor.
    """
    if value is None:
        return ""

    value = str(value).strip()

    # El CSV puede contener caracteres escapados procedentes de informes previos.
    value = value.replace("\\_", "_")

    return value


def parse_integer(value):
    """
    Convierte un valor a entero de forma robusta.
    """
    if value is None:
        return 0

    value = str(value).strip()

    if not value:
        return 0

    value = value.replace(",", "")

    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_float(value):
    """
    Convierte un valor a float de forma robusta.
    """
    if value is None:
        return 0.0

    value = str(value).strip()

    if not value:
        return 0.0

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_interaction(interaction):
    """
    Convierte:

        EDGE_LOCATION + NO_PREDICTION

    en:

        ["EDGE_LOCATION", "NO_PREDICTION"]
    """

    interaction = normalize_factor(interaction)

    if not interaction:
        return []

    parts = interaction.split(SEPARATOR)

    factors = []

    for part in parts:
        factor = normalize_factor(part)

        if factor and factor not in factors:
            factors.append(factor)

    return factors


# ============================================================================
# LOAD INPUT
# ============================================================================

def load_interactions():
    """
    Carga el CSV generado por:

    analyze_person_small_failure_interactions_v1.py

    Formato esperado:

        interaction,failures,percentage
    """

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "\nNo se encontró el CSV de interacciones:\n"
            f"{INPUT_CSV}\n\n"
            "Comprueba que primero hayas ejecutado:\n"
            "analyze_person_small_failure_interactions_v1.py"
        )

    rows = []

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "El CSV no contiene cabecera."
            )

        required_columns = {
            "interaction",
            "failures",
            "percentage"
        }

        missing = required_columns.difference(
            set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Faltan columnas obligatorias en el CSV: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            interaction = normalize_factor(
                row.get("interaction", "")
            )

            failures = parse_integer(
                row.get("failures", 0)
            )

            percentage = parse_float(
                row.get("percentage", 0)
            )

            if not interaction:
                continue

            if failures <= 0:
                continue

            factors = parse_interaction(interaction)

            if not factors:
                continue

            rows.append(
                {
                    "interaction": interaction,
                    "failures": failures,
                    "percentage": percentage,
                    "factors": factors,
                }
            )

    if not rows:
        raise ValueError(
            "El CSV se ha leído correctamente, "
            "pero no contiene interacciones válidas."
        )

    return rows


# ============================================================================
# TOTAL FAILURES
# ============================================================================

def calculate_total_failures(rows):
    """
    Calcula el total de small failures representados
    por el CSV de interacciones.

    Importante:
    El CSV de interacciones puede contener combinaciones de factores.
    Por tanto, aquí se suma cada interacción una sola vez.
    """

    return sum(
        row["failures"]
        for row in rows
    )


# ============================================================================
# FACTOR CONTRIBUTION
# ============================================================================

def calculate_factor_contribution(rows, total_failures):
    """
    Calcula para cada factor:

        - número de interacciones donde aparece
        - failures asociados
        - porcentaje sobre todos los failures
        - porcentaje sobre las interacciones
        - tamaño medio de interacción
    """

    factor_failures = defaultdict(int)
    factor_interaction_count = defaultdict(int)
    factor_occurrences = defaultdict(int)

    total_interactions = len(rows)

    for row in rows:

        failures = row["failures"]
        factors = row["factors"]

        for factor in factors:

            factor_failures[factor] += failures

            factor_interaction_count[factor] += 1

            factor_occurrences[factor] += 1

    results = []

    for factor in factor_failures:

        failures = factor_failures[factor]

        interaction_count = (
            factor_interaction_count[factor]
        )

        contribution_percentage = (
            failures / total_failures * 100
            if total_failures > 0
            else 0.0
        )

        interaction_percentage = (
            interaction_count / total_interactions * 100
            if total_interactions > 0
            else 0.0
        )

        results.append(
            {
                "factor": factor,
                "failures": failures,
                "contribution_percentage": contribution_percentage,
                "interaction_count": interaction_count,
                "interaction_percentage": interaction_percentage,
            }
        )

    results.sort(
        key=lambda x: (
            -x["failures"],
            x["factor"]
        )
    )

    return results


# ============================================================================
# FACTOR PAIR INTERACTIONS
# ============================================================================

def calculate_factor_interactions(rows):
    """
    Calcula las relaciones entre pares de factores.

    Ejemplo:

        EDGE_LOCATION + NO_PREDICTION

    genera:

        EDGE_LOCATION
        NO_PREDICTION

    y el par:

        EDGE_LOCATION + NO_PREDICTION
    """

    pair_failures = defaultdict(int)
    pair_occurrences = defaultdict(int)

    for row in rows:

        factors = sorted(
            set(row["factors"])
        )

        if len(factors) < 2:
            continue

        pairs = combinations(
            factors,
            2
        )

        for factor_a, factor_b in pairs:

            key = (
                factor_a,
                factor_b
            )

            pair_failures[key] += row["failures"]

            pair_occurrences[key] += 1

    results = []

    for (
        factor_a,
        factor_b
    ), failures in pair_failures.items():

        results.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "interaction": (
                    f"{factor_a}{SEPARATOR}{factor_b}"
                ),
                "failures": failures,
                "occurrences": pair_occurrences[
                    (factor_a, factor_b)
                ],
            }
        )

    results.sort(
        key=lambda x: (
            -x["failures"],
            x["factor_a"],
            x["factor_b"]
        )
    )

    return results


# ============================================================================
# FACTOR RANKING
# ============================================================================

def calculate_ranking(contribution_results):
    """
    Crea un ranking de factores.

    El ranking principal se basa en:

        failures asociados al factor
    """

    ranking = []

    for rank, item in enumerate(
        contribution_results,
        start=1
    ):

        ranking.append(
            {
                "rank": rank,
                "factor": item["factor"],
                "failures": item["failures"],
                "contribution_percentage": (
                    item["contribution_percentage"]
                ),
                "interaction_count": (
                    item["interaction_count"]
                ),
                "interaction_percentage": (
                    item["interaction_percentage"]
                ),
            }
        )

    return ranking


# ============================================================================
# WRITE CONTRIBUTION CSV
# ============================================================================

def write_contribution_csv(results):

    with OUTPUT_CONTRIBUTION.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "factor",
                "failures",
                "contribution_percentage",
                "interaction_count",
                "interaction_percentage",
            ]
        )

        for item in results:

            writer.writerow(
                [
                    item["factor"],
                    item["failures"],
                    f"{item['contribution_percentage']:.6f}",
                    item["interaction_count"],
                    f"{item['interaction_percentage']:.6f}",
                ]
            )


# ============================================================================
# WRITE RANKING CSV
# ============================================================================

def write_ranking_csv(ranking):

    with OUTPUT_RANKING.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "factor",
                "failures",
                "contribution_percentage",
                "interaction_count",
                "interaction_percentage",
            ]
        )

        for item in ranking:

            writer.writerow(
                [
                    item["rank"],
                    item["factor"],
                    item["failures"],
                    f"{item['contribution_percentage']:.6f}",
                    item["interaction_count"],
                    f"{item['interaction_percentage']:.6f}",
                ]
            )


# ============================================================================
# WRITE FACTOR INTERACTIONS CSV
# ============================================================================

def write_interactions_csv(results, total_failures):

    with OUTPUT_INTERACTIONS.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "factor_a",
                "factor_b",
                "interaction",
                "failures",
                "percentage_of_total_failures",
                "occurrences",
            ]
        )

        for item in results:

            percentage = (
                item["failures"]
                / total_failures
                * 100
                if total_failures > 0
                else 0.0
            )

            writer.writerow(
                [
                    item["factor_a"],
                    item["factor_b"],
                    item["interaction"],
                    item["failures"],
                    f"{percentage:.6f}",
                    item["occurrences"],
                ]
            )


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(
    rows,
    total_failures,
    contribution_results,
    interaction_results
):

    total_factors = len(
        contribution_results
    )

    top_factor = (
        contribution_results[0]
        if contribution_results
        else None
    )

    top_interaction = (
        interaction_results[0]
        if interaction_results
        else None
    )

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "========================================================================\n"
        )

        file.write(
            "SAR YOLO26 - PERSON SMALL FAILURE FACTOR CONTRIBUTION V1\n"
        )

        file.write(
            "========================================================================\n\n"
        )

        file.write(
            "INPUT\n"
        )

        file.write(
            "------------------------------------------------------------------------\n"
        )

        file.write(
            f"{INPUT_CSV}\n\n"
        )

        file.write(
            "OUTPUT\n"
        )

        file.write(
            "------------------------------------------------------------------------\n"
        )

        file.write(
            f"{OUTPUT_DIR}\n\n"
        )

        file.write(
            "========================================================================\n"
        )

        file.write(
            "RESULTADO PERSON SMALL FAILURE FACTOR CONTRIBUTION V1\n"
        )

        file.write(
            "========================================================================\n\n"
        )

        file.write(
            f"Interacciones analizadas: {len(rows):,}\n"
        )

        file.write(
            f"Small failures representados: {total_failures:,}\n"
        )

        file.write(
            f"Factores encontrados: {total_factors}\n"
        )

        file.write(
            f"Interacciones de factores: {len(interaction_results):,}\n\n"
        )

        # ------------------------------------------------------------
        # TOP FACTOR
        # ------------------------------------------------------------

        if top_factor:

            file.write(
                "FACTOR CON MAYOR CONTRIBUCIÓN\n"
            )

            file.write(
                "------------------------------------------------------------------------\n"
            )

            file.write(
                f"Factor: {top_factor['factor']}\n"
            )

            file.write(
                f"Failures: {top_factor['failures']:,}\n"
            )

            file.write(
                "Contribución: "
                f"{top_factor['contribution_percentage']:.2f}%\n"
            )

            file.write(
                f"Interacciones: {top_factor['interaction_count']:,}\n\n"
            )

        # ------------------------------------------------------------
        # TOP INTERACTION
        # ------------------------------------------------------------

        if top_interaction:

            file.write(
                "INTERACCIÓN DE FACTORES CON MAYOR CONTRIBUCIÓN\n"
            )

            file.write(
                "------------------------------------------------------------------------\n"
            )

            file.write(
                f"Interacción: {top_interaction['interaction']}\n"
            )

            file.write(
                f"Failures: {top_interaction['failures']:,}\n"
            )

            percentage = (
                top_interaction["failures"]
                / total_failures
                * 100
                if total_failures > 0
                else 0.0
            )

            file.write(
                f"Contribución: {percentage:.2f}%\n"
            )

            file.write(
                f"Ocurrencias: {top_interaction['occurrences']:,}\n\n"
            )

        # ------------------------------------------------------------
        # FACTOR RANKING
        # ------------------------------------------------------------

        file.write(
            "RANKING DE FACTORES\n"
        )

        file.write(
            "------------------------------------------------------------------------\n"
        )

        for index, item in enumerate(
            contribution_results,
            start=1
        ):

            file.write(
                f"{index:02d}. "
                f"{item['factor']:<35} "
                f"Failures={item['failures']:>7,} "
                f"Contribution="
                f"{item['contribution_percentage']:>7.2f}%\n"
            )

        file.write("\n")

        # ------------------------------------------------------------
        # TOP FACTOR PAIRS
        # ------------------------------------------------------------

        file.write(
            "TOP INTERACCIONES ENTRE FACTORES\n"
        )

        file.write(
            "------------------------------------------------------------------------\n"
        )

        for index, item in enumerate(
            interaction_results[:20],
            start=1
        ):

            percentage = (
                item["failures"]
                / total_failures
                * 100
                if total_failures > 0
                else 0.0
            )

            file.write(
                f"{index:02d}. "
                f"{item['interaction']:<65} "
                f"Failures={item['failures']:>7,} "
                f"{percentage:>7.2f}%\n"
            )

        file.write("\n")

        # ------------------------------------------------------------
        # INTERPRETATION
        # ------------------------------------------------------------

        file.write(
            "INTERPRETACIÓN\n"
        )

        file.write(
            "------------------------------------------------------------------------\n"
        )

        file.write(
            "Este análisis no interpreta un factor como una causa aislada "
            "del fallo. Un mismo failure puede pertenecer simultáneamente "
            "a varios factores dentro de una interacción.\n\n"
        )

        file.write(
            "Por tanto, las contribuciones de los factores pueden superar "
            "el 100% cuando se consideran conjuntamente. Esto es esperado "
            "y representa la naturaleza multivariable de los small-object "
            "detection failures.\n\n"
        )

        file.write(
            "El ranking debe utilizarse para identificar los factores "
            "que aparecen con mayor frecuencia dentro de los fallos "
            "residuales de personas pequeñas.\n\n"
        )

        file.write(
            "Las interacciones de factores permiten identificar "
            "combinaciones especialmente relevantes para posteriores "
            "experimentos de mejora del modelo.\n\n"
        )

        file.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE FACTOR CONTRIBUTION V1"
    )
    print("=" * 72)
    print()

    print("Input:")
    print(INPUT_CSV)
    print()

    print("Output:")
    print(OUTPUT_DIR)
    print()

    # ------------------------------------------------------------
    # CHECK INPUT
    # ------------------------------------------------------------

    print("Comprobando CSV de entrada...")

    if not INPUT_CSV.exists():

        print()
        print("[ERROR] No existe el CSV de entrada:")
        print(INPUT_CSV)
        print()

        raise FileNotFoundError(
            f"No se encontró:\n{INPUT_CSV}"
        )

    print("[OK] CSV encontrado.")
    print()

    # ------------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    print("Cargando interacciones...")

    rows = load_interactions()

    print(
        f"[OK] Interacciones cargadas: {len(rows):,}"
    )

    print()

    # ------------------------------------------------------------
    # TOTAL FAILURES
    # ------------------------------------------------------------

    total_failures = calculate_total_failures(
        rows
    )

    print(
        f"Small failures representados: "
        f"{total_failures:,}"
    )

    print()

    # ------------------------------------------------------------
    # FACTOR CONTRIBUTION
    # ------------------------------------------------------------

    print(
        "Calculando contribución de factores..."
    )

    contribution_results = (
        calculate_factor_contribution(
            rows,
            total_failures
        )
    )

    print(
        f"[OK] Factores encontrados: "
        f"{len(contribution_results)}"
    )

    print()

    # ------------------------------------------------------------
    # RANKING
    # ------------------------------------------------------------

    ranking = calculate_ranking(
        contribution_results
    )

    # ------------------------------------------------------------
    # FACTOR INTERACTIONS
    # ------------------------------------------------------------

    print(
        "Calculando interacciones entre factores..."
    )

    interaction_results = (
        calculate_factor_interactions(
            rows
        )
    )

    print(
        f"[OK] Interacciones de factores: "
        f"{len(interaction_results):,}"
    )

    print()

    # ------------------------------------------------------------
    # WRITE OUTPUTS
    # ------------------------------------------------------------

    print("Generando reports...")

    write_contribution_csv(
        contribution_results
    )

    print(
        f"[OK] {OUTPUT_CONTRIBUTION}"
    )

    write_ranking_csv(
        ranking
    )

    print(
        f"[OK] {OUTPUT_RANKING}"
    )

    write_interactions_csv(
        interaction_results,
        total_failures
    )

    print(
        f"[OK] {OUTPUT_INTERACTIONS}"
    )

    write_summary(
        rows,
        total_failures,
        contribution_results,
        interaction_results
    )

    print(
        f"[OK] {OUTPUT_SUMMARY}"
    )

    # ------------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON SMALL FAILURE FACTOR CONTRIBUTION V1"
    )
    print("=" * 72)
    print()

    print(
        f"Interacciones analizadas: {len(rows):,}"
    )

    print(
        f"Small failures:           {total_failures:,}"
    )

    print(
        f"Factores encontrados:      "
        f"{len(contribution_results)}"
    )

    print()

    print(
        "TOP FACTORES"
    )

    print(
        "------------------------------------------------------------------------"
    )

    for item in contribution_results[:15]:

        print(
            f"{item['factor']:<35} "
            f"Failures={item['failures']:>7,} "
            f"Contribution="
            f"{item['contribution_percentage']:>7.2f}%"
        )

    print()

    print(
        "TOP INTERACCIONES"
    )

    print(
        "------------------------------------------------------------------------"
    )

    for item in interaction_results[:15]:

        percentage = (
            item["failures"]
            / total_failures
            * 100
            if total_failures > 0
            else 0.0
        )

        print(
            f"{item['interaction']:<65} "
            f"Failures={item['failures']:>7,} "
            f"{percentage:>7.2f}%"
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