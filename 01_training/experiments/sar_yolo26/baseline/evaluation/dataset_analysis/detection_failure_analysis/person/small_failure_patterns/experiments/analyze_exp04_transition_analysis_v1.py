from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================================
# SAR YOLO26
# EXP04 - TRANSITION ANALYSIS V1
# ============================================================================
#
# OBJETIVO
# --------
# Explicar por qué EXP04 mejora a EXP01.
#
# Analiza:
#
#   FN -> TP
#   TP -> FN
#   TP -> TP
#   FN -> FN
#
# y cruza las transiciones con:
#
#   EXTREME_SMALL
#   DENSE_SCENE
#   CLOSE_NEIGHBORS
#   EDGE_LOCATION
#
# Además analiza interacciones:
#
#   EXTREME_SMALL + DENSE_SCENE
#   EXTREME_SMALL + CLOSE_NEIGHBORS
#   EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS
#   EDGE_LOCATION + EXTREME_SMALL
#   EDGE_LOCATION + DENSE_SCENE
#   EDGE_LOCATION + CLOSE_NEIGHBORS
#
# NO entrena.
# NO modifica dataset.
# NO modifica labels.
# NO modifica YAML.
#
# ============================================================================


# ============================================================================
# LOCALIZACIÓN
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_baseline_dir() -> Path:
    for parent in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:
        if parent.name.lower() == "baseline":
            return parent

    raise RuntimeError(
        "No se pudo localizar baseline."
    )


BASELINE_DIR = find_baseline_dir()


# ============================================================================
# INPUT
# ============================================================================

SOURCE_CSV = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp04_vs_exp01_factor_analysis_v1"
    / "reports"
    / "exp04_vs_exp01_objects_v1.csv"
)


# ============================================================================
# OUTPUT
# ============================================================================

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp04_transition_analysis_v1"
    / "reports"
)


TRANSITIONS_CSV = (
    REPORTS_DIR
    / "exp04_vs_exp01_transitions_v1.csv"
)

FACTORS_CSV = (
    REPORTS_DIR
    / "exp04_vs_exp01_transition_by_factor_v1.csv"
)

INTERACTIONS_CSV = (
    REPORTS_DIR
    / "exp04_vs_exp01_transition_by_interaction_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP04_VS_EXP01_TRANSITION_ANALYSIS_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_div(
    a: float,
    b: float,
) -> float:
    if b == 0:
        return 0.0

    return a / b


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# CARGA
# ============================================================================

def load_rows() -> list[dict]:

    if not SOURCE_CSV.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{SOURCE_CSV}"
        )

    with SOURCE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        fieldnames = set(
            reader.fieldnames or []
        )

    required = {
        "image",
        "gt_index",
        "EXTREME_SMALL",
        "EDGE_LOCATION",
        "CLOSE_NEIGHBORS",
        "DENSE_SCENE",
        "EXP01_TP",
        "EXP04_TP",
    }

    missing = (
        required
        -
        fieldnames
    )

    if missing:

        raise ValueError(
            "Faltan columnas:\n"
            +
            "\n".join(
                sorted(missing)
            )
        )

    return rows


# ============================================================================
# CLASIFICAR TRANSICIÓN
# ============================================================================

def get_transition(
    row: dict,
) -> str:

    exp01 = int(
        row["EXP01_TP"]
    )

    exp04 = int(
        row["EXP04_TP"]
    )

    if (
        exp01 == 0
        and
        exp04 == 1
    ):
        return "FN_TO_TP"

    if (
        exp01 == 1
        and
        exp04 == 0
    ):
        return "TP_TO_FN"

    if (
        exp01 == 1
        and
        exp04 == 1
    ):
        return "TP_TO_TP"

    return "FN_TO_FN"


# ============================================================================
# TRANSICIÓN GLOBAL
# ============================================================================

def build_global_transition_report(
    rows: list[dict],
) -> list[dict]:

    counts = defaultdict(int)

    for row in rows:

        transition = get_transition(
            row
        )

        counts[
            transition
        ] += 1

    total = len(
        rows
    )

    order = [
        "FN_TO_TP",
        "TP_TO_FN",
        "TP_TO_TP",
        "FN_TO_FN",
    ]

    output = []

    for transition in order:

        count = counts[
            transition
        ]

        output.append(
            {
                "transition":
                    transition,

                "count":
                    count,

                "percentage":
                    safe_div(
                        count,
                        total,
                    )
                    * 100.0,
            }
        )

    return output


# ============================================================================
# ANÁLISIS POR FACTOR
# ============================================================================

def build_factor_transition_report(
    rows: list[dict],
) -> list[dict]:

    factors = [
        "EXTREME_SMALL",
        "DENSE_SCENE",
        "CLOSE_NEIGHBORS",
        "EDGE_LOCATION",
    ]

    output = []

    for factor in factors:

        subset = [
            row
            for row in rows
            if int(
                row[factor]
            ) == 1
        ]

        if not subset:
            continue

        counts = defaultdict(int)

        for row in subset:

            counts[
                get_transition(row)
            ] += 1

        fn_to_tp = counts[
            "FN_TO_TP"
        ]

        tp_to_fn = counts[
            "TP_TO_FN"
        ]

        net = (
            fn_to_tp
            -
            tp_to_fn
        )

        output.append(
            {
                "factor":
                    factor,

                "gt":
                    len(subset),

                "FN_TO_TP":
                    fn_to_tp,

                "TP_TO_FN":
                    tp_to_fn,

                "TP_TO_TP":
                    counts[
                        "TP_TO_TP"
                    ],

                "FN_TO_FN":
                    counts[
                        "FN_TO_FN"
                    ],

                "net_recovery":
                    net,

                "recovery_rate":
                    safe_div(
                        fn_to_tp,
                        len(subset),
                    )
                    * 100.0,

                "loss_rate":
                    safe_div(
                        tp_to_fn,
                        len(subset),
                    )
                    * 100.0,
            }
        )

    return output


# ============================================================================
# INTERACCIONES
# ============================================================================

def build_interaction_report(
    rows: list[dict],
) -> list[dict]:

    interactions = [
        (
            "EXTREME_SMALL + DENSE_SCENE",
            [
                "EXTREME_SMALL",
                "DENSE_SCENE",
            ],
        ),

        (
            "EXTREME_SMALL + CLOSE_NEIGHBORS",
            [
                "EXTREME_SMALL",
                "CLOSE_NEIGHBORS",
            ],
        ),

        (
            "EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS",
            [
                "EXTREME_SMALL",
                "DENSE_SCENE",
                "CLOSE_NEIGHBORS",
            ],
        ),

        (
            "EDGE_LOCATION + EXTREME_SMALL",
            [
                "EDGE_LOCATION",
                "EXTREME_SMALL",
            ],
        ),

        (
            "EDGE_LOCATION + DENSE_SCENE",
            [
                "EDGE_LOCATION",
                "DENSE_SCENE",
            ],
        ),

        (
            "EDGE_LOCATION + CLOSE_NEIGHBORS",
            [
                "EDGE_LOCATION",
                "CLOSE_NEIGHBORS",
            ],
        ),
    ]

    output = []

    for name, factors in interactions:

        subset = [
            row
            for row in rows
            if all(
                int(row[factor]) == 1
                for factor in factors
            )
        ]

        if not subset:
            continue

        counts = defaultdict(int)

        for row in subset:

            counts[
                get_transition(row)
            ] += 1

        fn_to_tp = counts[
            "FN_TO_TP"
        ]

        tp_to_fn = counts[
            "TP_TO_FN"
        ]

        output.append(
            {
                "interaction":
                    name,

                "gt":
                    len(subset),

                "FN_TO_TP":
                    fn_to_tp,

                "TP_TO_FN":
                    tp_to_fn,

                "TP_TO_TP":
                    counts[
                        "TP_TO_TP"
                    ],

                "FN_TO_FN":
                    counts[
                        "FN_TO_FN"
                    ],

                "net_recovery":
                    fn_to_tp
                    -
                    tp_to_fn,

                "recovery_rate":
                    safe_div(
                        fn_to_tp,
                        len(subset),
                    )
                    * 100.0,

                "loss_rate":
                    safe_div(
                        tp_to_fn,
                        len(subset),
                    )
                    * 100.0,
            }
        )

    output.sort(
        key=lambda row:
            row["net_recovery"],
        reverse=True,
    )

    return output


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    rows: list[dict],
    transitions: list[dict],
    factors: list[dict],
    interactions: list[dict],
) -> None:

    global_counts = {
        row["transition"]:
            row["count"]
        for row in transitions
    }

    fn_to_tp = global_counts.get(
        "FN_TO_TP",
        0,
    )

    tp_to_fn = global_counts.get(
        "TP_TO_FN",
        0,
    )

    net = (
        fn_to_tp
        -
        tp_to_fn
    )

    best_factor = (
        max(
            factors,
            key=lambda row:
                row["net_recovery"],
        )
        if factors
        else None
    )

    best_interaction = (
        max(
            interactions,
            key=lambda row:
                row["net_recovery"],
        )
        if interactions
        else None
    )

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP04 VS EXP01 TRANSITION ANALYSIS V1",
        "=" * 72,
        "",
        "OBJETIVO",
        (
            "Explicar de dónde procede la mejora de EXP04 "
            "respecto a EXP01."
        ),
        "",
        "TRANSICIONES GLOBALES",
        "-" * 72,
        f"FN -> TP:      {fn_to_tp:,}",
        f"TP -> FN:      {tp_to_fn:,}",
        f"NET recovery:  {net:+d}",
        "",
        "LECTURA",
        "-" * 72,
    ]

    if net > 0:

        lines.append(
            (
                "EXP04 recupera más detecciones de las que pierde."
            )
        )

    elif net < 0:

        lines.append(
            (
                "EXP04 pierde más detecciones de las que recupera."
            )
        )

    else:

        lines.append(
            (
                "EXP04 tiene balance neto neutro."
            )
        )

    if best_factor is not None:

        lines.extend(
            [
                "",
                "MEJOR FACTOR",
                "-" * 72,
                (
                    f"{best_factor['factor']}"
                ),
                (
                    f"GT={best_factor['gt']:,}"
                ),
                (
                    f"FN->TP={best_factor['FN_TO_TP']:,}"
                ),
                (
                    f"TP->FN={best_factor['TP_TO_FN']:,}"
                ),
                (
                    f"Net={best_factor['net_recovery']:+d}"
                ),
            ]
        )

    if best_interaction is not None:

        lines.extend(
            [
                "",
                "MEJOR INTERACCIÓN",
                "-" * 72,
                (
                    f"{best_interaction['interaction']}"
                ),
                (
                    f"GT={best_interaction['gt']:,}"
                ),
                (
                    f"FN->TP={best_interaction['FN_TO_TP']:,}"
                ),
                (
                    f"TP->FN={best_interaction['TP_TO_FN']:,}"
                ),
                (
                    f"Net={best_interaction['net_recovery']:+d}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "INTERPRETACIÓN",
            "-" * 72,
            (
                "Los factores con mayor NET recovery son los candidatos "
                "más fuertes para explicar el mecanismo de mejora de EXP04."
            ),
            (
                "Un factor con muchos FN->TP pero también muchos TP->FN "
                "indica que EXP04 cambia fuertemente el comportamiento, "
                "pero no necesariamente de forma limpia."
            ),
            (
                "La información se utilizará para explicar EXP04; "
                "no se utilizará para modificar TEST_DEV."
            ),
            "",
            "IMPORTANTE: no se modificó dataset, labels ni YAML.",
        ]
    )

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - EXP04 VS EXP01 TRANSITION ANALYSIS V1"
    )
    print("=" * 72)

    rows = load_rows()

    print()
    print(
        f"[OK] Objetos SMALL: "
        f"{len(rows):,}"
    )

    transitions = (
        build_global_transition_report(
            rows
        )
    )

    factors = (
        build_factor_transition_report(
            rows
        )
    )

    interactions = (
        build_interaction_report(
            rows
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        TRANSITIONS_CSV,
        transitions,
    )

    write_csv(
        FACTORS_CSV,
        factors,
    )

    write_csv(
        INTERACTIONS_CSV,
        interactions,
    )

    build_summary(
        rows,
        transitions,
        factors,
        interactions,
    )

    print()
    print("=" * 72)
    print(
        "# TRANSICIONES EXP01 -> EXP04"
    )
    print("=" * 72)

    for row in transitions:

        print(
            f"{row['transition']:<10} "
            f"{row['count']:>7,} "
            f"{row['percentage']:>7.2f}%"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "# FACTORES"
    )
    print(
        "=" * 72
    )

    for row in factors:

        print(
            f"{row['factor']:<20} "
            f"GT={row['gt']:>7,} "
            f"FN->TP={row['FN_TO_TP']:>6,} "
            f"TP->FN={row['TP_TO_FN']:>6,} "
            f"NET={row['net_recovery']:+6d}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "# INTERACCIONES"
    )
    print(
        "=" * 72
    )

    for row in interactions:

        print(
            f"{row['interaction']:<50} "
            f"GT={row['gt']:>6,} "
            f"FN->TP={row['FN_TO_TP']:>5,} "
            f"TP->FN={row['TP_TO_FN']:>5,} "
            f"NET={row['net_recovery']:+5d}"
        )

    print()
    print(
        "REPORTS"
    )

    print(
        f"[OK] {TRANSITIONS_CSV}"
    )

    print(
        f"[OK] {FACTORS_CSV}"
    )

    print(
        f"[OK] {INTERACTIONS_CSV}"
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()
    print(
        "IMPORTANTE: dataset original NO modificado."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\n[CANCELADO]"
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "[ERROR EXP04 TRANSITION ANALYSIS]"
        )

        print(
            str(exc)
        )

        sys.exit(1)