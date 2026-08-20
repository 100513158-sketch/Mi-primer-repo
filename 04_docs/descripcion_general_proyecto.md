# SARC-Drone: descripción general del proyecto

## 1. Visión general

SARC-Drone es un proyecto de investigación y desarrollo orientado a construir un sistema de rescate asistido por visión por computadora y aeronaves autónomas o semiautónomas. La idea central es combinar aprendizaje automático, procesamiento de imágenes, análisis de datasets, exportación de modelos y una aplicación Android para convertir la detección visual en una herramienta útil para operaciones de búsqueda y rescate.

El proyecto no se limita a entrenar un modelo de detección. Su objetivo más amplio es crear un pipeline completo que vaya desde la adquisición y estandarización de datasets hasta la ejecución de un modelo optimizado en un entorno móvil o embebido, con una arquitectura pensada para evolucionar hacia integración con drones, IoT, nube y automatización.

## 2. Propósito del sistema

El sistema busca permitir que una plataforma visual pueda:

- identificar personas o objetos relevantes en imágenes o video capturadas desde drones;
- trabajar con múltiples datasets de dominio aéreo y de rescate;
- convertir el conocimiento aprendido en modelos de inferencia livianos y rápidos;
- desplegar esos modelos en aplicaciones Android;
- generar reportes, estadísticas y métricas para validar la calidad de los datos y los modelos;
- servir como base para futuras etapas de autonomía, integración con control remoto y despliegue en entornos operativos reales.

## 3. Alcance técnico del proyecto

SARC-Drone integra varias capas de trabajo:

1. Preparación y normalización de datasets.
2. Entrenamiento de modelos de detección.
3. Evaluación y validación de calidad.
4. Optimización y exportación de modelos a formatos como ONNX, OpenVINO y TFLite.
5. Integración con una aplicación Android.
6. Documentación, pruebas y trazabilidad de resultados.

El proyecto está pensado tanto para experimentación académica como para una base técnica de desarrollo aplicada.

## 4. Estructura general del repositorio

El repositorio está organizado en módulos funcionales:

- [README.md](../README.md): documentación principal y visión general del proyecto.
- [config.yaml](../config.yaml): configuración central del proyecto.
- [requirements.txt](../requirements.txt): dependencias de Python.
- [00_datasets](../00_datasets): almacenamiento, preparación y preparación de datasets.
- [01_training](../01_training): scripts de entrenamiento, conversión, validación, exportación y pipeline curricular.
- [02_models](../02_models): pesos de modelos, resultados y artefactos exportados.
- [03_android_app](../03_android_app): aplicación Android que consume o ejecuta los modelos.
- [04_docs](../04_docs): documentación técnica y guías operativas.
- [05_tests](../05_tests): validaciones, reportes y comprobaciones de release.

## 5. Componentes principales

### 5.1 Datasets y preparación de datos

La carpeta [00_datasets](../00_datasets) contiene el corazón del problema de datos. Aquí se almacenan:

- datasets crudos o sin procesar;
- datos procesados y convertidos al formato usado por el pipeline de entrenamiento;
- yaml de configuración por dataset;
- artefactos intermedios de preparación.

En este proyecto, los datasets no se tratan como archivos aislados, sino como parte del proceso de entrenamiento y validación. La calidad, estructura y consistencia de los datos son fundamentales, porque un modelo excelente no puede nacer de un dataset mal preparado.

### 5.2 Entrenamiento y pipeline curricular

La carpeta [01_training](../01_training) contiene scripts y artefactos para el ciclo de aprendizaje automático. Entre sus funciones principales están:

- preparar datasets para entrenar;
- verificar integridad y consistencia de labels e imágenes;
- generar archivos de configuración por dataset;
- ejecutar entrenamientos individuales y secuenciales;
- aplicar curriculum learning, transfiriendo conocimiento entre etapas;
- exportar modelos a distintos formatos.

Este enfoque permite que el sistema evolucione desde datasets más generales hacia datasets más específicos al dominio de rescate.

### 5.3 Modelos y artefactos

La carpeta [02_models](../02_models) concentra los resultados de entrenamiento y exportación:

- pesos en formato `.pt`;
- resultados de métricas en CSV;
- modelos exportados a ONNX, OpenVINO y TFLite;
- snapshots intermedios y comparaciones entre etapas.

La existencia de esta carpeta permite separar claramente lo que es entrenamiento experimental de lo que ya se considera un artefacto listo para evaluación o ejecución.

### 5.4 Aplicación Android

La carpeta [03_android_app](../03_android_app) contiene el frontend móvil del proyecto. La app se diseña para servir como capa de interacción, visualización y despliegue del modelo. Su responsabilidad es permitir que el usuario ejecute inferencia, vea resultados y eventualmente interactúe con el sistema en condiciones reales o simuladas.

Aunque el proyecto sigue evolucionando, esta capa es clave porque cierra el ciclo entre el desarrollo del modelo y la ejecución práctica del sistema.

### 5.5 Línea experimental SAR YOLO26

La investigación activa de entrenamiento y evaluación se encuentra en
`01_training/experiments/sar_yolo26/baseline/`. Esta línea parte de un baseline
de detección de personas y analiza cómo mejorar el recall de personas muy
pequeñas en imágenes aéreas.

La secuencia actual de experimentos va de EXP01 a EXP07 e incluye, entre otras
intervenciones, mayor resolución, crops de escenas densas, separación de
vecinos cercanos y crops próximos a los bordes. EXP07 combina una población de
personas extremadamente pequeñas, escenas densas y vecinos cercanos. Los
resultados se comparan con un protocolo constante sobre `test_dev`.

Los scripts de análisis y sus reportes están organizados dentro de
`01_training/experiments/sar_yolo26/baseline/evaluation/dataset_analysis/detection_failure_analysis/person/small_failure_patterns/`.
Entre los análisis actuales se encuentran la transición de detecciones de EXP04
frente a EXP01 y la evaluación de recall de EXP07.

Esta línea todavía es experimental. El modelo que finalmente se seleccione
deberá incorporarse al flujo Go/No-Go, de forma similar a la validación de la
variante inicial, antes de considerarlo candidato de release. Después de esa
selección se actualizará la exportación y la app Android. Mientras tanto, no se
debe tratar EXP07 como el modelo oficial de Android ni modificar los perfiles
Go/No-Go existentes por estos resultados parciales.

### 5.6 Documentación y pruebas

- [04_docs](../04_docs): reúne guías, procesos, flujos de comando y documentación operacional.
- [05_tests](../05_tests): contiene validaciones y reportes de verificación del pipeline y del proceso de release.

Esto ayuda a que el proyecto no solo funcione, sino que además sea reproducible, auditable y mantenible.

## 6. Flujo de trabajo general del proyecto

El flujo típico del proyecto es el siguiente:

1. Preparar o importar datasets.
2. Verificar su integridad y formato.
3. Limpiarlos, normalizarlos y convertirlos si es necesario.
4. Entrenar un modelo o una cadena de modelos secuenciales.
5. Evaluar métricas y calidad de datos.
6. Exportar el modelo a formatos útiles para inferencia.
7. Desplegarlo o integrarlo en la app Android.
8. Generar reportes y estadísticas para análisis posterior.

Este flujo es iterativo: un problema detectado en los datos o en la evaluación puede devolver el proceso a una fase anterior para corregirlo.

## 7. Enfoque de datasets y aprendizaje automático

El proyecto trabaja con múltiples datasets de detección y visión por computadora. Algunos son más generales, otros más específicos al dominio de rescate. Esa diversidad es una fortaleza, pero también exige un enfoque cuidadoso para:

- homogeneizar el formato de anotaciones;
- unificar o mapear clases;
- manejar diferencias entre datasets;
- filtrar relaciones irrelevantes o problemáticas;
- preservar información útil para la tarea de rescate.

La estrategia de entrenamiento se basa en la idea de pasar de una base general a una especialización más orientada a rescate.

## 8. Módulo nuevo: SAR_DATASET_STUDIO

Paralelamente al pipeline principal de entrenamiento, se está desarrollando un subproyecto llamado [00_datasets/SAR_DATASET_STUDIO](../00_datasets/SAR_DATASET_STUDIO). Este módulo representa una iniciativa más estructurada para trabajar con datasets de forma modular, con un pipeline propio para:

- importar datasets desde distintas fuentes;
- descargar datasets cuando sea posible;
- limpiar y validar datos;
- convertir formatos;
- generar estadísticas;
- producir reportes visuales y de calidad;
- registrar metadatos y manifiestos del proceso.

### 8.1 Objetivo de SAR_DATASET_STUDIO

Este subproyecto busca ser una capa de trabajo más explícita para la gestión de datos, separada del entrenamiento directo. Su propósito es convertir la manipulación de datasets en un proceso más ordenado, trazable y extensible.

### 8.2 Estructura general de SAR_DATASET_STUDIO

La estructura actual incluye carpetas y módulos de propósito claro:

- [00_datasets/SAR_DATASET_STUDIO/configs](../00_datasets/SAR_DATASET_STUDIO/configs): archivos de configuración como datasets, clases y pipeline.
- [00_datasets/SAR_DATASET_STUDIO/raw](../00_datasets/SAR_DATASET_STUDIO/raw): datos crudos organizados por dataset, como VisDrone, SeaDronesSee, RescueNet y FloodNet.
- [00_datasets/SAR_DATASET_STUDIO/processed](../00_datasets/SAR_DATASET_STUDIO/processed): resultados intermedios y etapas de procesamiento como imported, validated, cleaned, converted, merged y final.
- [00_datasets/SAR_DATASET_STUDIO/registry](../00_datasets/SAR_DATASET_STUDIO/registry): registro de datasets y manifiestos.
- [00_datasets/SAR_DATASET_STUDIO/reports](../00_datasets/SAR_DATASET_STUDIO/reports): reportes de descarga, validación, estadísticas, duplicados y calidad.
- [00_datasets/SAR_DATASET_STUDIO/logs](../00_datasets/SAR_DATASET_STUDIO/logs): trazabilidad y logs del flujo.
- [00_datasets/SAR_DATASET_STUDIO/src](../00_datasets/SAR_DATASET_STUDIO/src): código fuente organizado en módulos funcionales.
- [00_datasets/SAR_DATASET_STUDIO/tests](../00_datasets/SAR_DATASET_STUDIO/tests): pruebas del subproyecto.

### 8.3 Módulos incluidos dentro de src

El árbol de código fuente del subproyecto está pensado para separar responsabilidades por dominio:

- core: estructuras de datos comunes como Dataset, Annotation, BoundingBox, ImageRecord y excepciones.
- downloaders: lógica orientada a descargar datasets o preparar su adquisición.
- importers: módulos para importar datasets desde distintos formatos o fuentes.
- cleaners: procesos de limpieza y filtrado de datos.
- converters: conversiones entre formatos y representaciones.
- validators: validación de estructura, anotaciones y calidad general.
- viewers: visualización de muestras, anotaciones y análisis visual.
- statistics: generación de métricas y análisis estadístico.
- exporters: exportación de resultados y datasets preparados.
- assistants, utils, statistics y otros módulos: funciones auxiliares y extensiones del sistema.

### 8.4 Estado actual de SAR_DATASET_STUDIO

Este subproyecto está en fase inicial de construcción. La estructura base ya está definida, y se están creando los módulos y carpetas necesarias para sostener un pipeline completo de manejo de datos. Aunque aún no está totalmente integrado con el resto del pipeline de entrenamiento principal, su finalidad es clara: convertirse en una herramienta más robusta, modular y reutilizable para el manejo de datasets del proyecto.

## 9. Perspectiva de evolución del proyecto

A medida que el proyecto avance, se espera que SARC-Drone evolucione hacia una plataforma más completa con las siguientes capacidades:

- mejor integración entre datasets, entrenamiento y despliegue;
- más automatización en la validación de datos y métricas;
- mayor trazabilidad de resultados y decisiones de limpieza;
- soporte para más datasets y más tipos de anotaciones;
- integración con drones, control remoto y sistemas IoT;
- mejora en la experiencia de uso de la app Android.

## 10. Resumen ejecutivo

SARC-Drone es un proyecto integral que conecta visión por computadora, aprendizaje automático, procesamiento de datos y aplicaciones móviles en un mismo objetivo: construir un sistema útil para rescate asistido por inteligencia artificial. El trabajo principal está en la preparación de datos, el entrenamiento de modelos, la evaluación, la optimización y la ejecución en dispositivos móviles, con una evolución adicional hacia un sistema más amplio basado en datasets, análisis y operaciones de rescate.

La creación de SAR_DATASET_STUDIO refuerza esta visión al introducir una capa dedicada al manejo ordenado de datos, donde la calidad, la trazabilidad y la organización del flujo de datasets son tan importantes como el propio modelo.
