from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from sentinel_eye.stability.stability_module import ROI


@dataclass
class BoundingBox:
	x: int
	y: int
	w: int
	h: int
	area: int


@dataclass
class MotionMetrics:
	"""
	Métricas de detección de movimiento dentro de una ROI.
	- num_regions: cuántas regiones de movimiento se detectaron.
	- total_moving_area_ratio: área total en movimiento relativa al área de la ROI (0–1).
	- level: "NONE" | "LOW" | "HIGH".
	"""

	num_regions: int
	total_moving_area_ratio: float
	level: str


@dataclass
class MotionResult:
	metrics: MotionMetrics
	boxes: List[BoundingBox]


class MotionDetector:
	"""
	Detector de movimiento basado en diferencias entre frames/segmentación simple.
	"""

	def __init__(self, min_area_ratio: float = 0.001, max_area_ratio: float = 0.5) -> None:
		self.min_area_ratio = min_area_ratio
		self.max_area_ratio = max_area_ratio
		self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
			history=500,
			varThreshold=16,
			detectShadows=True,
		)

	def update(self, frame: np.ndarray, roi: Optional[ROI] = None, timestamp: Optional[float] = None) -> MotionResult:
		"""
		Procesa un frame (opcionalmente recortado a la ROI) y devuelve métricas y bounding boxes de movimiento.
		"""
		if roi is not None:
			x0, y0, w, h = roi.x, roi.y, roi.w, roi.h
			frame_roi = frame[y0 : y0 + h, x0 : x0 + w]
			roi_area = float(w * h)
		else:
			y0, x0 = 0, 0
			frame_roi = frame
			roi_area = float(frame.shape[0] * frame.shape[1])

		fg_mask = self.bg_subtractor.apply(frame_roi)

		_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
		fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

		contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		boxes: List[BoundingBox] = []
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			area = w * h
			if area < self.min_area_ratio * roi_area:
				continue
			if area > self.max_area_ratio * roi_area:
				continue
			global_x = x0 + x
			global_y = y0 + y
			boxes.append(BoundingBox(global_x, global_y, w, h, area))

		num_regions = len(boxes)
		total_moving_area = float(sum(box.area for box in boxes))
		total_moving_area_ratio = total_moving_area / roi_area if roi_area > 0 else 0.0
		level = self._level_from_ratio(total_moving_area_ratio)

		return MotionResult(
			metrics=MotionMetrics(
				num_regions=num_regions,
				total_moving_area_ratio=total_moving_area_ratio,
				level=level,
			),
			boxes=boxes,
		)

	def _level_from_ratio(self, ratio: float) -> str:
		if ratio < 0.001:
			return "NONE"
		if ratio < 0.01:
			return "LOW"
		return "HIGH"
