from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StabilityMetrics:
	"""
	Métricas de estabilidad entre frames consecutivos.
	- dx, dy: desplazamiento estimado (pixeles) entre el frame anterior y el actual.
	- mag: magnitud sqrt(dx^2 + dy^2).
	- accum_x, accum_y: desplazamiento acumulado desde el primer frame.
	- level: "STABLE" | "WOBBLE" | "DRIFT".
	- num_tracked: puntos de interés usados en la estimación.
	"""

	dx: float
	dy: float
	mag: float
	accum_x: float
	accum_y: float
	level: str
	num_tracked: int
	vibration_rms: float


@dataclass
class ROI:
	"""
	Región de interés rectangular.
	"""

	x: int
	y: int
	w: int
	h: int


class StabilityTracker:
	"""
	Estimador de estabilidad basado en flujo óptico dentro de una ROI central.
	"""

	def __init__(self, roi_fraction: float = 0.5) -> None:
		"""
		roi_fraction define el tamaño de la ROI central (0.5 => 50% del ancho/alto).
		"""
		self.roi_fraction = roi_fraction
		self.prev_gray: Optional[np.ndarray] = None
		self.prev_points: Optional[np.ndarray] = None  # shape (N, 1, 2)
		self.accum_x: float = 0.0
		self.accum_y: float = 0.0

		self.base_roi: Optional[ROI] = None
		self.current_roi: Optional[ROI] = None
		self.dx_window: deque[float] = deque(maxlen=30)
		self.dy_window: deque[float] = deque(maxlen=30)

		self.feature_params = dict(
			maxCorners=200,
			qualityLevel=0.01,
			minDistance=10,
			blockSize=7,
		)
		self.lk_params = dict(
			winSize=(21, 21),
			maxLevel=3,
			criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
		)

	def initialize(self, frame: np.ndarray) -> None:
		"""
		Inicializa el tracker:
		- Convierte a gris y detecta puntos de interés en la ROI central.
		- Define la ROI base centrada según roi_fraction.
		"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
		if self.prev_points is None:
			self.prev_points = np.empty((0, 1, 2), dtype=np.float32)
		self.prev_gray = gray

		h, w = gray.shape
		roi_w = int(w * self.roi_fraction)
		roi_h = int(h * self.roi_fraction)
		x = (w - roi_w) // 2
		# Sesga la ROI verticalmente hacia abajo (70% del espacio libre) para centrar en la calzada.
		y = int((h - roi_h) * 0.70)
		self.base_roi = ROI(x=x, y=y, w=roi_w, h=roi_h)
		self.current_roi = self.base_roi
		self.dx_window.clear()
		self.dy_window.clear()

	def update(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Tuple[StabilityMetrics, ROI]:
		"""
		Actualiza el tracker con un nuevo frame:
		- Si es el primer frame, llama a initialize.
		- Calcula dx, dy mediante flujo óptico en la ROI.
		- Actualiza desplazamientos acumulados y ROI compensada.
		- Devuelve (StabilityMetrics, ROI actual).
		"""
		if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
			self.initialize(frame)
			roi = self.current_roi or ROI(0, 0, frame.shape[1], frame.shape[0])
			level = self._level_from_motion(0.0, 0.0, 0.0)
			return StabilityMetrics(
				dx=0.0,
				dy=0.0,
				mag=0.0,
				accum_x=self.accum_x,
				accum_y=self.accum_y,
				level=level,
				num_tracked=int(len(self.prev_points) if self.prev_points is not None else 0),
				vibration_rms=0.0,
			), roi

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		next_points, status, err = cv2.calcOpticalFlowPyrLK(
			self.prev_gray,
			gray,
			self.prev_points,
			None,
			**self.lk_params,
		)

		if next_points is None or status is None:
			self.initialize(frame)
			roi = self.current_roi or ROI(0, 0, frame.shape[1], frame.shape[0])
			level = self._level_from_motion(0.0, 0.0, 0.0)
			return StabilityMetrics(
				dx=0.0,
				dy=0.0,
				mag=0.0,
				accum_x=self.accum_x,
				accum_y=self.accum_y,
				level=level,
				num_tracked=0,
				vibration_rms=0.0,
			), roi

		valid_mask = status.ravel() == 1
		valid_prev = self.prev_points[valid_mask]
		valid_next = next_points[valid_mask]
		num_valid = int(len(valid_prev))

		if num_valid < 10:
			self.initialize(frame)
			roi = self.current_roi or ROI(0, 0, frame.shape[1], frame.shape[0])
			level = self._level_from_motion(0.0, 0.0, 0.0)
			return StabilityMetrics(
				dx=0.0,
				dy=0.0,
				mag=0.0,
				accum_x=self.accum_x,
				accum_y=self.accum_y,
				level=level,
				num_tracked=num_valid,
				vibration_rms=0.0,
			), roi

		dx_i = valid_next[:, 0, 0] - valid_prev[:, 0, 0]
		dy_i = valid_next[:, 0, 1] - valid_prev[:, 0, 1]

		dx = float(np.median(dx_i))
		dy = float(np.median(dy_i))
		self.dx_window.append(dx)
		self.dy_window.append(dy)
		self.accum_x += dx
		self.accum_y += dy
		mag = float(np.hypot(dx, dy))
		vibration_rms = self._motion_rms()
		drift_mag = float(np.hypot(self.accum_x, self.accum_y))

		h, w = gray.shape
		if self.current_roi is None:
			self.current_roi = self.base_roi or ROI(0, 0, w, h)
		new_x = self.current_roi.x - int(round(dx))
		new_y = self.current_roi.y - int(round(dy))
		new_x = max(0, min(new_x, w - self.current_roi.w))
		new_y = max(0, min(new_y, h - self.current_roi.h))
		self.current_roi = ROI(new_x, new_y, self.current_roi.w, self.current_roi.h)

		self.prev_gray = gray
		self.prev_points = valid_next.reshape(-1, 1, 2)

		level = self._level_from_motion(mag, vibration_rms, drift_mag)
		return StabilityMetrics(
			dx=dx,
			dy=dy,
			mag=mag,
			accum_x=self.accum_x,
			accum_y=self.accum_y,
			level=level,
			num_tracked=num_valid,
			vibration_rms=vibration_rms,
		), self.current_roi

	def _motion_rms(self) -> float:
		"""Calcula RMS de desplazamientos recientes para detectar vibración constante."""
		if not self.dx_window or not self.dy_window:
			return 0.0
		arr_dx = np.array(self.dx_window, dtype=np.float32)
		arr_dy = np.array(self.dy_window, dtype=np.float32)
		rms = np.sqrt(np.mean(arr_dx * arr_dx + arr_dy * arr_dy))
		return float(rms)

	def _level_from_motion(self, mag: float, vibration_rms: float, drift_mag: float) -> str:
		"""
		Clasifica el movimiento en:
		- STABLE: casi sin movimiento ni vibración.
		- VIBRATION: mucha oscilación pero poco drift acumulado.
		- WOBBLE: sacudidas moderadas.
		- DRIFT: desplazamiento neto elevado.
		"""
		if drift_mag > 8.0 or mag >= 2.5:
			return "DRIFT"
		if vibration_rms > 1.4 and drift_mag < 8.0:
			return "VIBRATION"
		if mag >= 0.5 or vibration_rms > 0.8:
			return "WOBBLE"
		return "STABLE"

	def get_base_roi(self) -> Optional[ROI]:
		"""Devuelve la ROI base respecto al frame de referencia."""
		return self.base_roi
