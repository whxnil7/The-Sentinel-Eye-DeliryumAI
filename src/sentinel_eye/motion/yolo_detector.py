from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from sentinel_eye.stability.stability_module import ROI


@dataclass
class YoloDetection:
	x: int
	y: int
	w: int
	h: int
	cls_id: int
	cls_name: str
	score: float


class YoloObjectDetector:
	"""
	Wrapper simple para detección con un modelo YOLO en formato ONNX.
	"""

	def __init__(self, model_path: str = "models/yolov8n.onnx", conf_th: float = 0.35, iou_th: float = 0.45) -> None:
		self.model_path = model_path
		self.conf_th = conf_th
		self.iou_th = iou_th
		self.img_size = 640
		self.class_names = {
			0: "person",
			1: "bicycle",
			2: "car",
			3: "motorcycle",
			5: "bus",
			7: "truck",
		}
		self.vehicle_class_ids = {2, 3, 5, 7}

		self.session: ort.InferenceSession = ort.InferenceSession(
			self.model_path,
			providers=["CPUExecutionProvider"],
		)
		print(f"[YOLO] Cargando modelo ONNX desde: {model_path}")
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name

	def detect(self, frame: np.ndarray, roi: Optional[ROI] = None) -> List[YoloDetection]:
		"""
		Realiza detección de objetos sobre un frame BGR.
		Si se proporciona ROI, opera sobre la región recortada y devuelve coordenadas globales.
		"""
		if roi is not None:
			roi_img = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
			offset_x, offset_y = roi.x, roi.y
			work_img = roi_img
		else:
			work_img = frame
			offset_x, offset_y = 0, 0

		blob, scale_x, scale_y = self._preprocess(work_img)
		preds = self.session.run([self.output_name], {self.input_name: blob})[0]
		detections = self._postprocess(preds, scale_x, scale_y, offset_x, offset_y)
		if (len(detections)) > 0:
			print(f"[YOLO] detecciones en este frame: {len(detections)}")

		if detections:
			best = max(detections, key=lambda d: d.score)
			print(
				f"[YOLO] mejor detección: {best.cls_name} score={best.score:.2f} "
				f"bbox=({best.x},{best.y},{best.w},{best.h})"
			)
		return detections

	def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
		"""
		Convierte a RGB, redimensiona a tamaño fijo, normaliza y devuelve tensor (1,3,H,W)
		junto con los factores de escala para mapear de vuelta a la imagen original.
		"""
		h0, w0 = img.shape[:2]
		img_rgb = img[:, :, ::-1]
		img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
		blob = img_resized.astype(np.float32) / 255.0
		blob = np.transpose(blob, (2, 0, 1))
		blob = np.expand_dims(blob, 0)
		scale_x = w0 / float(self.img_size)
		scale_y = h0 / float(self.img_size)
		return blob, scale_x, scale_y

	def _postprocess(self, preds: np.ndarray, scale_x: float, scale_y: float, offset_x: int, offset_y: int) -> List[YoloDetection]:
		"""
		Convierte la salida de YOLO (1, N, 85) en detecciones en coordenadas del frame completo.
		"""
		preds = np.asarray(preds, dtype=np.float32)
		if preds.ndim == 3:
			preds = preds[0]
		dets: List[YoloDetection] = []

		for row in preds:
			box = row[0:4]
			obj_conf = float(row[4])
			cls_probs = row[5:]
			cls_id = int(np.argmax(cls_probs))
			cls_conf = float(cls_probs[cls_id])
			score = obj_conf * cls_conf

			if score < self.conf_th:
				continue
			if cls_id not in self.vehicle_class_ids:
				continue

			cx, cy, w_box, h_box = box
			x1 = (cx - w_box / 2.0) * scale_x
			y1 = (cy - h_box / 2.0) * scale_y
			x2 = (cx + w_box / 2.0) * scale_x
			y2 = (cy + h_box / 2.0) * scale_y

			x1 += offset_x
			x2 += offset_x
			y1 += offset_y
			y2 += offset_y

			x1 = max(0, int(x1))
			y1 = max(0, int(y1))
			x2 = max(0, int(x2))
			y2 = max(0, int(y2))

			w = x2 - x1
			h = y2 - y1
			if w <= 0 or h <= 0:
				continue

			cls_name = self.class_names.get(cls_id, str(cls_id))
			dets.append(
				YoloDetection(
					x=int(x1),
					y=int(y1),
					w=int(w),
					h=int(h),
					cls_id=cls_id,
					cls_name=cls_name,
					score=float(score),
				)
			)

		return self._nms(dets)

	def _nms(self, detections: List[YoloDetection]) -> List[YoloDetection]:
		if not detections:
			return []
		dets = sorted(detections, key=lambda d: d.score, reverse=True)
		keep: List[YoloDetection] = []

		while dets:
			current = dets.pop(0)
			keep.append(current)
			dets = [d for d in dets if self._iou(current, d) <= self.iou_th]

		return keep

	def _iou(self, a: YoloDetection, b: YoloDetection) -> float:
		ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
		bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h

		ix1 = max(ax1, bx1)
		iy1 = max(ay1, by1)
		ix2 = min(ax2, bx2)
		iy2 = min(ay2, by2)

		inter_w = max(0, ix2 - ix1)
		inter_h = max(0, iy2 - iy1)
		inter_area = inter_w * inter_h

		area_a = a.w * a.h
		area_b = b.w * b.h
		union = area_a + area_b - inter_area
		if union <= 0:
			return 0.0
		return inter_area / union
