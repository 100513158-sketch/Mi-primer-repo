from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class VisDroneVisualizer:
    """
    Visualizador de anomalías del dataset VisDrone.

    Genera para cada anomalía:

        full.jpg
            Imagen completa con las anotaciones.

        crop.jpg
            Recorte ampliado alrededor de la anomalía.

    El punto problemático se marca con una cruz roja y NO se coloca
    ninguna etiqueta encima de la zona para facilitar la inspección.
    """

    # =========================================================
    # CLASES OFICIALES VISDRONE
    # =========================================================

    CLASSES = {
        0: "ignored_region",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
        11: "others",
    }

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(self, dataset_root: Path):

        self.dataset_root = Path(
            dataset_root
        ).resolve()

        # dataset_root:
        #
        # SAR_DATASET_STUDIO/
        # └── raw/
        #     └── VisDrone/
        #         └── original/
        #

        self.project_root = (
            self.dataset_root.parents[2]
        )

        # Nunca guardar resultados dentro de raw.

        self.output_root = (
            self.project_root
            / "reports"
            / "validation"
            / "visual"
            / "anomalies"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # FUENTE
    # =========================================================

    def get_font(self, size: int = 18):

        try:

            return ImageFont.truetype(
                "arial.ttf",
                size,
            )

        except Exception:

            return ImageFont.load_default()

    # =========================================================
    # BUSCAR IMAGEN
    # =========================================================

    def find_image(
        self,
        annotation_file: Path,
    ) -> Path | None:

        image_stem = annotation_file.stem

        extensions = [
            ".jpg",
            ".JPG",
            ".jpeg",
            ".JPEG",
            ".png",
            ".PNG",
        ]

        # -----------------------------------------------------
        # Buscar sustituyendo annotations -> images
        # -----------------------------------------------------

        parts = list(
            annotation_file.parts
        )

        try:

            index = parts.index(
                "annotations"
            )

            image_parts = parts[:index]

            image_parts.append(
                "images"
            )

            image_parts.extend(
                parts[index + 1:]
            )

            image_directory = Path(
                *image_parts
            )

            for extension in extensions:

                candidate = (
                    image_directory
                    / f"{image_stem}{extension}"
                )

                if candidate.exists():

                    return candidate

        except ValueError:

            pass

        # -----------------------------------------------------
        # Búsqueda alternativa
        # -----------------------------------------------------

        for extension in extensions:

            matches = list(
                self.dataset_root.rglob(
                    f"{image_stem}{extension}"
                )
            )

            if matches:

                return matches[0]

        return None

    # =========================================================
    # LEER ANNOTATIONS
    # =========================================================

    def read_annotations(
        self,
        annotation_file: Path,
    ) -> list[dict]:

        annotations = []

        text = annotation_file.read_text(
            encoding="utf-8"
        )

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            # -------------------------------------------------
            # VisDrone:
            #
            # x,y,width,height,score,class,truncation,occlusion
            #
            # Puede existir coma final.
            # -------------------------------------------------

            clean_line = line.rstrip(",")

            parts = [
                p.strip()
                for p in clean_line.split(",")
            ]

            if len(parts) != 8:
                continue

            try:

                x = int(parts[0])
                y = int(parts[1])
                width = int(parts[2])
                height = int(parts[3])
                score = int(parts[4])
                class_id = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])

            except ValueError:

                continue

            annotations.append(
                {
                    "line": line_number,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "score": score,
                    "class_id": class_id,
                    "class_name": self.CLASSES.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "raw": line,
                }
            )

        return annotations

    # =========================================================
    # DIBUJAR TODAS LAS ANOTACIONES
    # =========================================================

    def draw_annotations(
        self,
        image: Image.Image,
        annotations: list[dict],
        highlight_line: int | None = None,
    ):

        draw = ImageDraw.Draw(
            image
        )

        font = self.get_font(18)

        for annotation in annotations:

            x = annotation["x"]
            y = annotation["y"]

            width = annotation["width"]
            height = annotation["height"]

            x2 = x + width
            y2 = y + height

            class_id = annotation["class_id"]
            class_name = annotation["class_name"]

            line_number = annotation["line"]

            # -------------------------------------------------
            # IGNORAR REGIONES
            # -------------------------------------------------

            if class_id == 0:

                # Las regiones ignoradas se muestran con
                # línea amarilla discontinua simulada.

                if width > 0 and height > 0:

                    draw.rectangle(
                        [
                            x,
                            y,
                            x2,
                            y2,
                        ],
                        outline="yellow",
                        width=2,
                    )

                continue

            # -------------------------------------------------
            # BOX VÁLIDA
            # -------------------------------------------------

            if width > 0 and height > 0:

                if line_number == highlight_line:

                    draw.rectangle(
                        [
                            x,
                            y,
                            x2,
                            y2,
                        ],
                        outline="red",
                        width=5,
                    )

                else:

                    draw.rectangle(
                        [
                            x,
                            y,
                            x2,
                            y2,
                        ],
                        outline="lime",
                        width=2,
                    )

                # -------------------------------------------------
                # Etiqueta SOLO para cajas suficientemente grandes.
                #
                # Esto evita llenar de texto las imágenes.
                # -------------------------------------------------

                if width >= 15 and height >= 15:

                    label = (
                        f"{class_name} "
                        f"[{class_id}]"
                    )

                    self.draw_label(
                        draw,
                        x,
                        y,
                        label,
                        font,
                    )

            # -------------------------------------------------
            # BOX INVÁLIDA
            # -------------------------------------------------

            else:

                size = 12

                draw.line(
                    [
                        x - size,
                        y - size,
                        x + size,
                        y + size,
                    ],
                    fill="red",
                    width=5,
                )

                draw.line(
                    [
                        x - size,
                        y + size,
                        x + size,
                        y - size,
                    ],
                    fill="red",
                    width=5,
                )

        return image

    # =========================================================
    # ETIQUETA
    # =========================================================

    def draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        font,
    ):

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        width = (
            bbox[2] - bbox[0]
        )

        height = (
            bbox[3] - bbox[1]
        )

        text_x = max(
            0,
            x,
        )

        text_y = max(
            0,
            y - height - 6,
        )

        draw.rectangle(
            [
                text_x,
                text_y,
                text_x + width + 8,
                text_y + height + 6,
            ],
            fill="black",
        )

        draw.text(
            (
                text_x + 4,
                text_y + 3,
            ),
            text,
            fill="white",
            font=font,
        )

    # =========================================================
    # CREAR CRUZ DE ANOMALÍA
    # =========================================================

    def draw_cross(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        size: int = 15,
        width: int = 4,
    ):

        draw.line(
            [
                x - size,
                y - size,
                x + size,
                y + size,
            ],
            fill="red",
            width=width,
        )

        draw.line(
            [
                x - size,
                y + size,
                x + size,
                y - size,
            ],
            fill="red",
            width=width,
        )

        # Círculo alrededor del punto

        draw.ellipse(
            [
                x - size - 4,
                y - size - 4,
                x + size + 4,
                y + size + 4,
            ],
            outline="red",
            width=3,
        )

    # =========================================================
    # CREAR CROP
    # =========================================================

    def create_crop(
        self,
        image: Image.Image,
        annotation: dict,
        output_file: Path,
        crop_size: int = 250,
        scale: int = 4,
    ):

        x = annotation["x"]
        y = annotation["y"]

        image_width = image.width
        image_height = image.height

        half = crop_size // 2

        # -----------------------------------------------------
        # Coordenadas del crop
        # -----------------------------------------------------

        left = max(
            0,
            x - half,
        )

        top = max(
            0,
            y - half,
        )

        right = min(
            image_width,
            x + half,
        )

        bottom = min(
            image_height,
            y + half,
        )

        crop = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        # -----------------------------------------------------
        # Escalar
        # -----------------------------------------------------

        new_width = (
            crop.width * scale
        )

        new_height = (
            crop.height * scale
        )

        crop = crop.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.NEAREST,
        )

        draw = ImageDraw.Draw(
            crop
        )

        # -----------------------------------------------------
        # Coordenadas del punto dentro del crop
        # -----------------------------------------------------

        local_x = (
            x - left
        ) * scale

        local_y = (
            y - top
        ) * scale

        # -----------------------------------------------------
        # Cruz
        # -----------------------------------------------------

        self.draw_cross(
            draw,
            int(local_x),
            int(local_y),
            size=20,
            width=5,
        )

        # -----------------------------------------------------
        # Texto informativo en una banda superior
        # -----------------------------------------------------

        font = self.get_font(20)

        info = (
            f"LINE {annotation['line']} | "
            f"CLASS {annotation['class_id']} "
            f"({annotation['class_name']}) | "
            f"BOX {annotation['width']}x"
            f"{annotation['height']} | "
            f"SCORE {annotation['score']}"
        )

        # Banda negra superior

        draw.rectangle(
            [
                0,
                0,
                crop.width,
                35,
            ],
            fill="black",
        )

        draw.text(
            (
                8,
                8,
            ),
            info,
            fill="white",
            font=font,
        )

        # -----------------------------------------------------
        # Guardar
        # -----------------------------------------------------

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        crop.save(
            output_file,
            quality=95,
        )

    # =========================================================
    # VISUALIZAR UNA ANOMALÍA
    # =========================================================

    def visualize_anomaly(
        self,
        annotation_file: Path,
        line_number: int,
        anomaly_id: str,
    ) -> bool:

        print()
        print("=" * 70)

        print(
            f"Analizando: "
            f"{annotation_file.name}"
        )

        print(
            f"Línea problemática: "
            f"{line_number}"
        )

        # -----------------------------------------------------
        # Buscar imagen
        # -----------------------------------------------------

        image_file = self.find_image(
            annotation_file
        )

        if image_file is None:

            print(
                "[ERROR] No se encontró "
                "la imagen."
            )

            return False

        print(
            f"Imagen: {image_file}"
        )

        # -----------------------------------------------------
        # Leer annotations
        # -----------------------------------------------------

        annotations = self.read_annotations(
            annotation_file
        )

        # -----------------------------------------------------
        # Buscar annotation problemática
        # -----------------------------------------------------

        problematic = None

        for annotation in annotations:

            if (
                annotation["line"]
                == line_number
            ):

                problematic = annotation

                break

        if problematic is None:

            print(
                "[ERROR] No se encontró "
                "la línea problemática."
            )

            return False

        print()
        print(
            "Annotation problemática:"
        )

        print(
            problematic["raw"]
        )

        print(
            f"Clase: "
            f"{problematic['class_id']} "
            f"({problematic['class_name']})"
        )

        print(
            f"X={problematic['x']} "
            f"Y={problematic['y']} "
            f"W={problematic['width']} "
            f"H={problematic['height']}"
        )

        print(
            f"Score={problematic['score']}"
        )

        # -----------------------------------------------------
        # Abrir imagen
        # -----------------------------------------------------

        try:

            image = Image.open(
                image_file
            ).convert("RGB")

        except Exception as exc:

            print(
                f"[ERROR] No se pudo abrir: "
                f"{exc}"
            )

            return False

        print(
            f"Resolución: "
            f"{image.width}x{image.height}"
        )

        # -----------------------------------------------------
        # Crear directorio
        # -----------------------------------------------------

        output_directory = (
            self.output_root
            / anomaly_id
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # FULL
        # =====================================================

        full_image = image.copy()

        self.draw_annotations(
            full_image,
            annotations,
            highlight_line=line_number,
        )

        # -----------------------------------------------------
        # En la imagen completa dibujamos la cruz encima
        # de la anomalía.
        # -----------------------------------------------------

        full_draw = ImageDraw.Draw(
            full_image
        )

        self.draw_cross(
            full_draw,
            problematic["x"],
            problematic["y"],
            size=15,
            width=4,
        )

        full_output = (
            output_directory
            / "full.jpg"
        )

        full_image.save(
            full_output,
            quality=95,
        )

        print()
        print(
            f"[OK] Full:"
        )

        print(
            full_output
        )

        # =====================================================
        # CROP
        # =====================================================

        crop_output = (
            output_directory
            / "crop.jpg"
        )

        self.create_crop(
            image=image,
            annotation=problematic,
            output_file=crop_output,
            crop_size=250,
            scale=4,
        )

        print()
        print(
            f"[OK] Crop:"
        )

        print(
            crop_output
        )

        return True

    # =========================================================
    # EJECUTAR ANOMALÍAS CONOCIDAS
    # =========================================================

    def visualize_known_anomalies(self):

        anomalies = [

            {
                "id": "anomaly_01",

                "annotation": (
                    self.dataset_root
                    / "train"
                    / "VisDrone2019-DET-train"
                    / "annotations"
                    / "0000293_03401_d_0000939.txt"
                ),

                "line": 130,
            },

            {
                "id": "anomaly_02",

                "annotation": (
                    self.dataset_root
                    / "train"
                    / "VisDrone2019-DET-train"
                    / "annotations"
                    / "9999985_00000_d_0000020.txt"
                ),

                "line": 12,
            },

            {
                "id": "anomaly_03",

                "annotation": (
                    self.dataset_root
                    / "train"
                    / "VisDrone2019-DET-train"
                    / "annotations"
                    / "9999999_00590_d_0000267.txt"
                ),

                "line": 89,
            },
        ]

        print()
        print("=" * 70)
        print(
            "VISDRONE ANOMALY VISUALIZER"
        )
        print("=" * 70)

        print()
        print(
            "Dataset:"
        )

        print(
            self.dataset_root
        )

        print()
        print(
            "Salida:"
        )

        print(
            self.output_root
        )

        success = 0

        # -----------------------------------------------------
        # Procesar
        # -----------------------------------------------------

        for anomaly in anomalies:

            annotation_file = (
                anomaly["annotation"]
            )

            if not annotation_file.exists():

                print()
                print(
                    "[ERROR] No existe:"
                )

                print(
                    annotation_file
                )

                continue

            result = self.visualize_anomaly(
                annotation_file=annotation_file,
                line_number=anomaly["line"],
                anomaly_id=anomaly["id"],
            )

            if result:

                success += 1

        # -----------------------------------------------------
        # Resumen
        # -----------------------------------------------------

        print()
        print("=" * 70)

        print(
            f"Anomalías procesadas: "
            f"{success}/{len(anomalies)}"
        )

        print()
        print(
            "Resultados:"
        )

        print(
            self.output_root
        )

        print("=" * 70)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # Obtener SAR_DATASET_STUDIO
    #
    # __file__:
    #
    # SAR_DATASET_STUDIO/
    # └── src/
    #     └── viewers/
    #         └── visdrone_visualizer.py
    #
    # parents[2] = SAR_DATASET_STUDIO
    # ---------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dataset_root = (
        project_root
        / "raw"
        / "VisDrone"
        / "original"
    )

    if not dataset_root.exists():

        print()
        print(
            "[ERROR] Dataset no encontrado:"
        )

        print(
            dataset_root
        )

        raise SystemExit(1)

    visualizer = VisDroneVisualizer(
        dataset_root
    )

    visualizer.visualize_known_anomalies()

