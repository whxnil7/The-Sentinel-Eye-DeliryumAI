from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from sentinel_eye.config import ROI, build_activity_rois


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
	activity_rois: List[ROI]


class MotionDetector:
	"""
	Detector de movimiento basado en diferencias entre frames/segmentación simple.
	"""

	def __init__(self, min_area_ratio: float = 0.001, max_area_ratio: float = 0.5, motion_stride: int = 2) -> None:
		self.min_area_ratio = min_area_ratio
		self.max_area_ratio = max_area_ratio
		self.motion_stride = max(1, motion_stride)
		self.frame_count: int = 0
		self._last_result: Optional[MotionResult] = None
		self.activity_rois: Optional[List[ROI]] = None
		self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
			history=500,
			varThreshold=16,
			detectShadows=True,
		)

	def update(self, frame: np.ndarray, roi: Optional[ROI] = None, timestamp: Optional[float] = None) -> MotionResult:
		"""
		Procesa un frame (opcionalmente recortado a la ROI) y devuelve métricas y bounding boxes de movimiento.
		"""
		self.frame_count += 1
		if self._last_result is not None and self.frame_count % self.motion_stride != 0:
			return self._last_result

		h, w = frame.shape[:2]
		if self.activity_rois is None:
			self.activity_rois = build_activity_rois(w, h)

		if roi is not None:
			rois_to_process: List[ROI] = [roi]
		else:
			rois_to_process = self.activity_rois or []

		all_boxes: List[BoundingBox] = []
		total_area = 0.0
		total_roi_area = 0.0

		for r in rois_to_process:
			x0, y0, rw, rh = r.x, r.y, r.w, r.h
			frame_roi = frame[y0 : y0 + rh, x0 : x0 + rw]
			roi_area = float(rw * rh)
			if roi_area <= 0:
				continue

			fg_mask = self.bg_subtractor.apply(frame_roi)

			_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
			fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

			contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for contour in contours:
				bx, by, bw, bh = cv2.boundingRect(contour)
				area = bw * bh
				if area < self.min_area_ratio * roi_area:
					continue
				if area > self.max_area_ratio * roi_area:
					continue
				global_x = x0 + bx
				global_y = y0 + by
				all_boxes.append(BoundingBox(global_x, global_y, bw, bh, area))
				total_area += area

			total_roi_area += roi_area

		num_regions = len(all_boxes)
		total_moving_area_ratio = total_area / total_roi_area if total_roi_area > 0 else 0.0
		level = self._level_from_ratio(total_moving_area_ratio)

		result = MotionResult(
			metrics=MotionMetrics(
				num_regions=num_regions,
				total_moving_area_ratio=total_moving_area_ratio,
				level=level,
			),
			boxes=all_boxes,
			activity_rois=self.activity_rois or [],
		)
		self._last_result = result
		return result

	def _level_from_ratio(self, ratio: float) -> str:
		if ratio < 0.001:
			return "NONE"
		if ratio < 0.01:
			return "LOW"
		return "HIGH"
