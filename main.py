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
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	pipeline = VideoPipeline(
		video_source=args.video_path,
		output_path=args.output_video,
		show_gui=not args.no_gui,
	)
	pipeline.run()


if __name__ == "__main__":
	main()
