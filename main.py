import argparse

from sentinel_eye.pipeline import VideoPipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="The Sentinel Eye - Video Processor")
	parser.add_argument(
		"--video-path",
		type=str,
		default="data/raw/video_prueba_1.avi",
		help="Ruta al video de entrada (archivo local).",
	)
	parser.add_argument(
		"--output-video",
		type=str,
		default="data/output/output_raw.mp4",
		help="Ruta al video de salida con overlays.",
	)
	parser.add_argument(
		"--no-gui",
		action="store_true",
		help="Si se especifica, no se abre ventana de visualización (modo headless/Docker).",
	)
	parser.add_argument(
		"--yolo-model",
		type=str,
		default="models/yolo11n.onnx",
		help="Ruta al modelo YOLO ONNX a usar para detección.",
	)
	parser.add_argument(
		"--yolo-conf",
		type=float,
		default=0.28,
		help="Umbral de confianza para YOLO.",
	)
	parser.add_argument(
		"--yolo-iou",
		type=float,
		default=0.5,
		help="Umbral IoU para NMS de YOLO.",
	)
	parser.add_argument(
		"--yolo-stride",
		type=int,
		default=3,
		help="Procesa YOLO cada N frames para acelerar (frame skipping).",
	)
	parser.add_argument(
		"--yolo-allow-all-classes",
		action="store_true",
		help="Si se especifica, YOLO acepta todas las clases en lugar de solo vehículos.",
	)
	parser.add_argument(
		"--no-yolo-quantize",
		action="store_true",
		help="Desactiva uso/creación de modelo cuantizado INT8 (ONNX Runtime).",
	)
	parser.add_argument(
		"--yolo-threads",
		type=int,
		default=0,
		help="Número de hilos intra-op/inter-op para ONNX Runtime (0 = auto).",
	)
	parser.add_argument(
		"--preview-real-time",
		action="store_true",
		help="Fuerza la vista previa a ir a velocidad de fps original (solo aplica si hay GUI).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	pipeline = VideoPipeline(
		video_source=args.video_path,
		output_path=args.output_video,
		show_gui=not args.no_gui,
		yolo_model_path=args.yolo_model,
		yolo_conf=args.yolo_conf,
		yolo_iou=args.yolo_iou,
		yolo_allow_all_classes=args.yolo_allow_all_classes,
		yolo_stride=args.yolo_stride,
		yolo_use_quantized=not args.no_yolo_quantize,
		yolo_num_threads=args.yolo_threads or None,
		preview_real_time=args.preview_real_time,
	)
	pipeline.run()


if __name__ == "__main__":
	main()
