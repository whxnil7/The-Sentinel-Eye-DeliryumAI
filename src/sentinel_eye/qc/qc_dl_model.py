from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class DLQCOutput:
	p_blur_bad: float
	p_occlusion_bad: float
	p_light_bad: float


class DeepQCEstimator:
	"""
	Estimador ligero de calidad de imagen basado en DL.

	Si no existe el modelo ONNX, is_available() será False y predict()
	devuelve None. Así el resto del sistema puede seguir usando solo
	las heurísticas clásicas sin romperse.
	"""

	def __init__(self, model_path: str = "models/qc_light.onnx") -> None:
		self.model_path = model_path
		self.session: Optional[ort.InferenceSession] = None
		self.input_name: Optional[str] = None

		try:
			self.session = ort.InferenceSession(
				self.model_path,
				providers=["CPUExecutionProvider"],
			)
			self.input_name = self.session.get_inputs()[0].name
			print(f"[DL-QC] Modelo ONNX cargado desde: {self.model_path}")
		except Exception:
			# No spameamos el error; simplemente dejamos el DL desactivado.
			print(f"[DL-QC] Modelo ONNX no disponible ({self.model_path}), usando solo heurísticas.")
			self.session = None
			self.input_name = None

	def is_available(self) -> bool:
		return self.session is not None and self.input_name is not None

	def predict(self, frame: np.ndarray) -> Optional[DLQCOutput]:
		"""
		Devuelve probabilidades de problemas de calidad o None si no hay modelo.

		Suponemos que el modelo ONNX espera entradas (1, 3, 128, 128) normalizadas
		en [0, 1] y entrega un vector (1, 3) con sigmoids:
		[p_blur_bad, p_occlusion_bad, p_light_bad].
		"""
		if not self.is_available():
			return None

		# Preprocesar: redimensionar a 128x128, BGR->RGB, normalizar
		img = cv2.resize(frame, (128, 128))
		img = img[:, :, ::-1]  # BGR -> RGB
		img = img.astype(np.float32) / 255.0
		img = np.transpose(img, (2, 0, 1))  # CHW
		img = np.expand_dims(img, 0)        # NCHW

		inputs = {self.input_name: img}
		outputs = self.session.run(None, inputs)

		probs = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
		if probs.shape[0] != 3:
			print(f"[DL-QC] Salida inesperada del modelo: shape={probs.shape}")
			return None

		return DLQCOutput(
			p_blur_bad=float(probs[0]),
			p_occlusion_bad=float(probs[1]),
			p_light_bad=float(probs[2]),
		)
