from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class VideoPipeline:
	"""
	Pipeline principal que:
	- Lee frames desde un video.
	- Llama a los módulos de QC, estabilidad y detección de movimiento.
	- Muestra los frames (por ahora sin overlays complejos).
	"""

	def __init__(self, video_source: str) -> None:
		self.video_source: str = video_source
		self.cap: Optional[cv2.VideoCapture] = None

		# TODO: inyectar instancias de módulos cuando los implementemos
		self.qc_module = None
		self.stability_module = None
		self.motion_module = None

	def _open_capture(self) -> None:
		self.cap = cv2.VideoCapture(self.video_source)
		if not self.cap.isOpened():
			raise RuntimeError(f"No se pudo abrir la fuente de video: {self.video_source}")

	def run(self) -> None:
		self._open_capture()
		assert self.cap is not None

		frame_idx = 0
		t0 = time.time()
		processed_frames = 0

		while True:
			ret, frame = self.cap.read()
			if not ret:
				break

			# En esta etapa solo pasamos el frame tal cual.
			# Más adelante:
			# - frame = self._run_qc(frame)
			# - frame = self._run_stability(frame)
			# - frame = self._run_motion_detection(frame)

			self._display_frame(frame, frame_idx)

			processed_frames += 1
			frame_idx += 1

			# Salir con la tecla 'q'
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		t1 = time.time()
		elapsed = max(t1 - t0, 1e-6)
		fps = processed_frames / elapsed
		print(f"[VideoPipeline] Frames procesados: {processed_frames} | FPS ~ {fps:.2f}")

		self._release()

	def _display_frame(self, frame: np.ndarray, frame_idx: int) -> None:
		"""
		Por ahora solo mostramos el frame y un texto con el índice.
		Luego aquí dibujaremos QC_Score, gráficos de vibración y bounding boxes.
		"""
		cv2.putText(
			frame,
			f"Frame {frame_idx}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)
		cv2.imshow("The Sentinel Eye - raw", frame)

	def _release(self) -> None:
		if self.cap is not None:
			self.cap.release()
		cv2.destroyAllWindows()
