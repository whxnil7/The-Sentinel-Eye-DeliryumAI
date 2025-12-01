import argparse

from sentinel_eye.pipeline import VideoPipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="The Sentinel Eye - Video Processor")
	parser.add_argument(
		"--video-path",
		type=str,
		default="data/raw/video_prueba.mp4",
		help="Ruta al video de entrada (archivo local).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	pipeline = VideoPipeline(video_source=args.video_path)
	pipeline.run()


if __name__ == "__main__":
	main()
