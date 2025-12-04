# The Sentinel Eye
Backend de pre-procesamiento para vigilancia en mina (blur/polvo/luz, vibración, detección optimizada).

## Cómo correr
- Local: `pip install -r requirements.txt` y luego `python main.py --no-gui --output-video data/output/output_raw.mp4`.
- Docker Compose (reproducible): `docker-compose up --build` genera `data/output/output_compose.mp4` y `data/output/qc_metrics.csv` montados en el host.
- Dashboard: abre `data/output/dashboard.html` y carga `qc_metrics.csv` para graficar QC/estabilidad/movimiento.

## Módulos clave
- **QC Score (0-100)**: varianza de Laplaciano, brillo, contraste, densidad de bordes y ratio de lente tapada; alertas por blur/occlusión/luz con refuerzo de modelo ONNX ligero (`models/qc_light.onnx`).
- **Estabilidad / Self-healing**: flujo óptico en ROI central, RMS de vibración, nivel STABLE/WOBBLE/VIBRATION/DRIFT; ROI compensada contra deriva y panel de trazas, dibuja ROI base vs ajustada.
- **Movimiento**: sustracción de fondo en ROIs de actividad, nivel NONE/LOW/HIGH y bounding boxes.
- **Detección optimizada**: YOLO ONNX vía ONNX Runtime con cuantización INT8 automática (`models/yolo11n.int8.onnx`), inferencia en batch sobre múltiples ROIs, frame skipping configurable (`--yolo-stride`) y clipping a la unión de ROIs para maximizar FPS.

## Artefactos de salida
- Video con overlays: `data/output/output_*.mp4`.
- Métricas/alertas: `data/output/qc_metrics.csv` (QC/estabilidad/movimiento, incluye `stab_vibration_rms`).

## Flags útiles
- `--yolo-stride N` controla salto de frames para YOLO.
- `--no-yolo-quantize` desactiva la versión INT8 si quieres forzar FP32.
- `--yolo-threads K` ajusta hilos de ONNX Runtime (intra/inter).
- `--preview-real-time` mantiene la velocidad del video original en modo GUI.
