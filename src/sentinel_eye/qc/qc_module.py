from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from sentinel_eye.qc.qc_dl_model import DLQCOutput, DeepQCEstimator


@dataclass
class QCMetrics:
	"""
	Contenedor de métricas de calidad de imagen en escala 0–100.
	"""

	blur_score: float
	brightness_score: float
	contrast_score: float
	global_score: float
	var_laplacian: float
	mean_brightness: float
	sat_ratio: float
	std_gray: float
	occlusion_score: float
	edge_density: float
	blocked_ratio: float
	details: Optional[Dict[str, float]] = None


@dataclass
class QCAlerts:
	"""
	Alertas y niveles categorizados por métrica de QC.
	"""

	blur_level: str          # OK | WARN | CRITICAL
	occlusion_level: str     # OK | WARN | CRITICAL
	light_level: str         # OK | WARN | CRITICAL

	blur_alert: bool         # blur_score < threshold
	occlusion_alert: bool    # occlusion_score < threshold (placeholder)
	light_alert: bool        # light_score < threshold


@dataclass
class QCStatus:
	"""
	Niveles categorizados para cada métrica de QC.
	"""

	blur: str
	brightness: str
	contrast: str
	global_level: str


class ImageQualityAssessor:
	"""
	Evalúa la calidad de un frame de video (nitidez, brillo, contraste)
	con métricas clásicas (sin modelos de deep learning).
	"""

	BRIGHTNESS_LOW = 80.0  # rango “óptimo” inferior de brillo medio
	BRIGHTNESS_HIGH = 180.0  # rango “óptimo” superior de brillo medio
	SAT_PENALTY_MULT = 2.0  # multiplica el ratio de saturación para penalizar glare

	def __init__(self) -> None:
		# Límites de varianza de Laplaciano para mapear blur a 0–100
		self.blur_var_min = 150.0
		self.blur_var_max = 350.0
		# Límites de std para mapear contraste a 0–100
		self.contrast_std_min = 25.0
		self.contrast_std_max = 70.0
		# Pesos para el score global
		self.weight_blur = 0.4
		self.weight_brightness = 0.2
		self.weight_contrast = 0.4
		# Umbrales categóricos
		self.category_ok = 75.0
		self.category_warn = 60.0
		# Umbrales específicos para heurísticas
		self.blur_critical_th = 40.0
		self.contrast_critical_th = 35.0
		self.low_light_mean_th = 60.0
		self.glare_sat_th = 0.01  # 1% píxeles saturados
		# Modelo ligero opcional de DL para alertas
		self.dl_estimator = DeepQCEstimator()  # intentará cargar models/qc_light.onnx
		self.dl_threshold = 0.7  # probabilidad para considerar "malo"

	def evaluate(self, frame: np.ndarray, timestamp: Optional[float] = None) -> tuple[QCMetrics, QCAlerts]:
		"""
		Recibe un frame BGR de OpenCV y produce métricas en escala 0–100.
		Aplica:
		- Varianza del Laplaciano para blur/enfoque.
		- Brillo medio con penalización por sub/sobre-exposición y glare.
		- Desviación estándar como proxy de contraste.
		"""
		gray = self._to_gray(frame)

		var_lap = self._variance_of_laplacian(gray)
		mean_brightness = float(gray.mean())
		sat_ratio = float((gray > 240).mean())
		std_gray = float(gray.std())
		edge_density, blocked_ratio, low_texture_ratio = self._texture_metrics(gray)

		blur_score = self._estimate_blur(var_lap)
		brightness_score = self._estimate_brightness(mean_brightness, sat_ratio)
		contrast_score = self._estimate_contrast(std_gray)
		occlusion_score = self._estimate_occlusion(edge_density, blocked_ratio, low_texture_ratio)
		global_score = self._aggregate_scores(blur_score, brightness_score, contrast_score)
		# Penaliza el score global si la lente parece tapada.
		global_score *= max(0.0, 1.0 - blocked_ratio * 0.35)
		global_score = float(np.clip(global_score, 0.0, 100.0))

		metrics = QCMetrics(
			blur_score=blur_score,
			brightness_score=brightness_score,
			contrast_score=contrast_score,
			global_score=global_score,
			var_laplacian=var_lap,
			mean_brightness=mean_brightness,
			sat_ratio=sat_ratio,
			std_gray=std_gray,
			occlusion_score=occlusion_score,
			edge_density=edge_density,
			blocked_ratio=blocked_ratio,
			details={
				"low_texture_ratio": low_texture_ratio,
				"edge_density": edge_density,
				"blocked_ratio": blocked_ratio,
			},
		)

		# --- Blurring / Desenfoque ---
		blur_level = self._level_from_score(metrics.blur_score, self.category_ok, self.category_warn)
		blur_alert = metrics.blur_score < self.category_warn

		# --- Oclusión / Suciedad ---
		occlusion_level = self._level_from_score(metrics.occlusion_score, self.category_ok, self.category_warn)
		occlusion_alert = (
			metrics.occlusion_score < self.category_warn
			or metrics.blocked_ratio > 0.25
			or metrics.details.get("low_texture_ratio", 0.0) > 0.6  # type: ignore[union-attr]
		)

		# --- Low Light / Glare ---
		# Low light: brillo medio muy bajo
		# Glare: muchos píxeles saturados o contraste lavado con brillo alto
		light_score = metrics.brightness_score
		low_light_bad = metrics.mean_brightness < self.low_light_mean_th
		glare_bad = metrics.sat_ratio > self.glare_sat_th and metrics.contrast_score < self.contrast_critical_th

		if low_light_bad or glare_bad:
			light_level = "CRITICAL"
		else:
			light_level = self._level_from_score(light_score, self.category_ok, self.category_warn)

		light_alert = low_light_bad or glare_bad or light_score < self.category_warn

		dl_output: Optional[DLQCOutput] = None
		if self.dl_estimator.is_available():
			dl_output = self.dl_estimator.predict(frame)

		if dl_output is not None:
			# Combinar heurística + DL (OR conservador)
			if dl_output.p_blur_bad >= self.dl_threshold:
				blur_alert = True
				blur_level = "CRITICAL"

			if dl_output.p_occlusion_bad >= self.dl_threshold:
				occlusion_alert = True
				occlusion_level = "CRITICAL"

			if dl_output.p_light_bad >= self.dl_threshold:
				light_alert = True
				light_level = "CRITICAL"
			if metrics.details is not None:
				metrics.details["dl_p_blur_bad"] = dl_output.p_blur_bad
				metrics.details["dl_p_occlusion_bad"] = dl_output.p_occlusion_bad
				metrics.details["dl_p_light_bad"] = dl_output.p_light_bad

		alerts = QCAlerts(
			blur_level=blur_level,
			occlusion_level=occlusion_level,
			light_level=light_level,
			blur_alert=blur_alert,
			occlusion_alert=occlusion_alert,
			light_alert=light_alert,
		)

		return metrics, alerts

	def classify(self, qc: QCMetrics) -> QCStatus:
		"""
		Clasifica cada score en niveles OK / WARN / CRITICAL.
		"""
		blur_level = self._level_from_score(qc.blur_score)
		bright_level = self._level_from_score(qc.brightness_score)
		contrast_level = self._level_from_score(qc.contrast_score)
		global_level = self._level_from_score(qc.global_score)
		return QCStatus(
			blur=blur_level,
			brightness=bright_level,
			contrast=contrast_level,
			global_level=global_level,
		)

	def _to_gray(self, frame: np.ndarray) -> np.ndarray:
		return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	def _variance_of_laplacian(self, gray: np.ndarray) -> float:
		lap = cv2.Laplacian(gray, cv2.CV_64F)
		return float(lap.var())

	def _estimate_blur(self, var_lap: float) -> float:
		"""
		Usa varianza del Laplaciano; valores pequeños indican desenfoque.
		Normaliza entre umbrales min/max y satura en 0–100.
		"""
		return self._score_from_variance(var_lap)

	def _estimate_brightness(self, mean_brightness: float, sat_ratio: float) -> float:
		"""
		Penaliza si el brillo medio está fuera de [BRIGHTNESS_LOW, BRIGHTNESS_HIGH]
		y si hay exceso de saturación (glare).
		"""
		return self._score_from_brightness(
			mean_brightness,
			sat_ratio,
			low=self.BRIGHTNESS_LOW,
			high=self.BRIGHTNESS_HIGH,
		)

	def _estimate_contrast(self, std_gray: float) -> float:
		"""
		Usa la desviación estándar de la luminancia como proxy de contraste.
		Desvíos muy bajos indican imagen lavada.
		"""
		return self._score_from_contrast(std_gray)

	def _estimate_occlusion(self, edge_density: float, blocked_ratio: float, low_texture_ratio: float) -> float:
		"""
		Detecta lente tapada/polvo combinando densidad de bordes y áreas bloqueadas.
		- edge_density: porcentaje de píxeles que son bordes (Canny) en [0,1].
		- blocked_ratio: porcentaje de píxeles muy oscuros o saturados.
		- low_texture_ratio: proporción de celdas con muy poca varianza.
		"""
		texture_score = edge_density * 100.0
		penalty = blocked_ratio * 120.0 + low_texture_ratio * 60.0
		score = max(0.0, texture_score - penalty)
		return float(np.clip(score, 0.0, 100.0))

	def _aggregate_scores(self, blur: float, brightness: float, contrast: float) -> float:
		"""
		Combina las métricas individuales en un score global (0–100).
		"""
		return float(
			self.weight_blur * blur
			+ self.weight_brightness * brightness
			+ self.weight_contrast * contrast
		)

	def _score_from_variance(self, var_value: float) -> float:
		"""
		Mapea la varianza del Laplaciano a 0–100 usando umbrales min/max.
		- var <= min -> 0
		- var >= max -> 100
		- en el rango se interpola linealmente.
		"""
		if var_value <= self.blur_var_min:
			return 0.0
		if var_value >= self.blur_var_max:
			return 100.0
		span = max(self.blur_var_max - self.blur_var_min, 1e-6)
		score = (var_value - self.blur_var_min) / span * 100.0
		return float(np.clip(score, 0.0, 100.0))

	def _score_from_brightness(self, mean_brightness: float, sat_ratio: float, low: float, high: float) -> float:
		score = 100.0

		if mean_brightness < low:
			under = (low - mean_brightness) / max(low, 1e-6)
			score *= max(0.0, 1.0 - under)
		elif mean_brightness > high:
			over = (mean_brightness - high) / max(255.0 - high, 1e-6)
			score *= max(0.0, 1.0 - over)

		score *= max(0.0, 1.0 - sat_ratio * self.SAT_PENALTY_MULT)
		return float(np.clip(score, 0.0, 100.0))

	def _score_from_contrast(self, std_value: float) -> float:
		"""
		Mapea la desviación estándar de luminancia a 0–100.
		- std <= min -> 0 (poco contraste / lavado)
		- std >= max -> 100 (buen contraste, saturado al techo)
		"""
		if std_value <= self.contrast_std_min:
			return 0.0
		if std_value >= self.contrast_std_max:
			return 100.0
		span = max(self.contrast_std_max - self.contrast_std_min, 1e-6)
		score = (std_value - self.contrast_std_min) / span * 100.0
		return float(np.clip(score, 0.0, 100.0))

	def _level_from_score(self, score: float, ok_th: float = 80.0, warn_th: float = 50.0) -> str:
		"""
		Convierte un score 0–100 en nivel categórico.
		"""
		if score >= ok_th:
			return "OK"
		if score >= warn_th:
			return "WARN"
		return "CRITICAL"

	def _texture_metrics(self, gray: np.ndarray) -> tuple[float, float, float]:
		"""
		Calcula densidad de bordes (Canny), ratio de bloqueo (pixeles casi negros o saturados)
		y proporción de celdas con poca textura para detectar polvo/oclusión.
		"""
		edges = cv2.Canny(gray, 80, 150)
		edge_density = float(edges.mean() / 255.0)

		blocked_mask = (gray < 20) | (gray > 235)
		blocked_ratio = float(blocked_mask.mean())

		grid = 4
		h, w = gray.shape
		step_h = max(1, h // grid)
		step_w = max(1, w // grid)
		low_texture = 0
		total_cells = 0
		for i in range(grid):
			for j in range(grid):
				cell = gray[i * step_h : min(h, (i + 1) * step_h), j * step_w : min(w, (j + 1) * step_w)]
				if cell.size == 0:
					continue
				total_cells += 1
				if cell.std() < 12.0:
					low_texture += 1
		low_texture_ratio = float(low_texture) / float(total_cells or 1)
		return edge_density, blocked_ratio, low_texture_ratio
