from pathlib import Path
from collections import Counter

ROOT = Path(r"processed\converted\VisDrone")

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
    10: "others",
}

splits = ["train", "val", "test_dev"]

global_counter = Counter()

print("=" * 80)
print("ANÁLISIS DE CLASES VISDRONE")
print("=" * 80)

for split in splits:

    labels_dir = ROOT / split / "labels"
    counter = Counter()

    for label_file in labels_dir.glob("*.txt"):

        for line in label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])
            except ValueError:
                continue

            counter[cls] += 1

    global_counter.update(counter)

    total = sum(counter.values())

    print()
    print(f"[{split}]")
    print("-" * 80)

    for cls in sorted(CLASS_NAMES):
        count = counter[cls]
        percentage = (count / total * 100) if total else 0

        print(
            f"{cls:>2} | "
            f"{CLASS_NAMES[cls]:<20} | "
            f"{count:>8} | "
            f"{percentage:>6.2f}%"
        )

    print(f"\nTOTAL OBJETOS: {total:,}")

print()
print("=" * 80)
print("DISTRIBUCIÓN GLOBAL")
print("=" * 80)

total_global = sum(global_counter.values())

for cls in sorted(CLASS_NAMES):

    count = global_counter[cls]
    percentage = count / total_global * 100

    print(
        f"{cls:>2} | "
        f"{CLASS_NAMES[cls]:<20} | "
        f"{count:>8} | "
        f"{percentage:>6.2f}%"
    )

print()
print(f"TOTAL GLOBAL: {total_global:,}")

print()
print("=" * 80)
print("PROPUESTA INICIAL PARA SARC-DRONE")
print("=" * 80)

print("""
0 = person
1 = vehicle
""")

person = global_counter[0] + global_counter[1]

vehicle = (
    global_counter[2]
    + global_counter[3]
    + global_counter[4]
    + global_counter[5]
    + global_counter[6]
    + global_counter[7]
    + global_counter[8]
    + global_counter[9]
)

print(f"PERSON  : {person:,}")
print(f"VEHICLE : {vehicle:,}")

print()
print("NOTA:")
print("La clase 'others' no se incluye inicialmente.")
print("El dataset original NO ha sido modificado.")
print("=" * 80)
