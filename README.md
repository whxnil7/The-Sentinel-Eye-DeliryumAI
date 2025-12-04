# The Sentinel Eye
Backend de pre-procesamiento para vigilancia en mina (blur/polvo/luz, vibración, detección optimizada) realizado por Nilsson Ignacio Acevedo Peñaloza para la prueba técnica de Deliryum.AI.

## Arquitectura del pipeline
- Loop secuencial (OpenCV + ONNX Runtime) en `src/sentinel_eye/pipeline.py`.
- **QC (`qc_module`)**: scores 0–100 usando Laplaciano, brillo, contraste, densidad de bordes, ratio de lente tapada; alerta con modelo ligero ONNX (`models/qc_light.onnx`).
- **Estabilidad (`stability_module`)**: flujo óptico (Lucas-Kanade) en ROI central; calcula dx/dy, vibración RMS y mueve la ROI compensada.
- **Movimiento (`motion_module`)**: MOG2 sobre la ROI activa; reporta #regiones y área relativa.
- **Detección**: YOLO ONNX (`yolo11n.onnx`) con opción INT8; frame skipping configurable (`--yolo-stride`); clipping a la ROI activa.
- Salidas: video con overlays y `data/output/qc_metrics.csv`; dashboard estático en `data/output/dashboard.html`.
- Modular: cada componente (QC, estabilidad, movimiento, detección) vive en su propio módulo y se inyecta en el pipeline. Puedes reemplazar/añadir módulos (p. ej. otro detector, un tracker, un filtro de clima) sin cambiar el resto del flujo; basta con implementar la misma interfaz de entrada/salida y enchufarlo en `pipeline.py`.

## Cómo correr
- Local: `pip install -r requirements.txt` y luego `python main.py --no-gui --output-video data/output/output_raw.mp4`.
- Docker Compose: `docker-compose up --build` genera `data/output/output_compose.mp4` y `data/output/qc_metrics.csv`.
- Dashboard: abrir `data/output/dashboard.html` y cargar `qc_metrics.csv`.

## Flags útiles
- `--yolo-stride N` (salto de frames YOLO), `--no-yolo-quantize` (forzar FP32), `--yolo-threads K` (hilos ORT), `--preview-real-time` (sincronizar preview al fps original).

## Rendimiento
- CPU Docker (stride=4, sin GUI): ~13 FPS registrados.
- Local GUI (stride=3, fallback FP32): ~12–14 FPS.
- Con cuantización activa: se espera +15–25% FPS en CPU.

## Futuro / mejoras
- Modelos de detección especializados en polvo/niebla o distillation para edge.
- ROI dinámica guiada por heatmaps de movimiento o detecciones previas.
- Multi-device MLOps: métricas periódicas, PSI/KS para drift de blur/occlusión, canary rollout y feature flags.
- Tracking ligero para reducir falsas alarmas y mejorar consistencia de cajas.
- Aceleración: TensorRT/CoreML en dispositivos que lo soporten; batching real si el modelo lo permite.

## Por qué falló el modelo cuantizado (INT8)
Se intenta cargar `models/yolo11n.int8.onnx` y cae a FP32 con el error:
`[ONNXRuntimeError] NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) ...`.
Posibles razones:
- El proveedor de ejecución activo (CPUExecutionProvider) no soporta la op ConvInteger usada en el modelo INT8.
- El ONNX Runtime instalado no incluye kernels de cuantización necesarios o se compiló sin aceleración para INT8.
- El modelo cuantizado se generó con opset/operadores no soportados por la build actual.

### Cómo mitigar
- Forzar FP32 con `--no-yolo-quantize` (ya cae automáticamente si falla).
- Instalar/usar una build de ORT con soporte de INT8 en la plataforma (p.ej. `onnxruntime-gpu` o `onnxruntime` con aceleración adecuada).
- Regenerar el modelo cuantizado con un pipeline compatible (opset y kernels disponibles).
