from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from sentinel_eye.stability.stability_module import ROI

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:
    QuantType = None  # type: ignore
    quantize_dynamic = None  # type: ignore


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

    def __init__(
        self,
        model_path: str = "models/yolo11n.onnx",
        conf_th: float = 0.28,
        iou_th: float = 0.5,
        allowed_class_ids: Optional[set[int]] = None,
        use_quantized: bool = True,
        quantized_model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
    ) -> None:
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
        self.allowed_class_ids = allowed_class_ids if allowed_class_ids is not None else {2, 3, 5, 7}
        self.use_quantized = use_quantized
        self.quantized_model_path = quantized_model_path or self._default_quantized_path(model_path)
        self.num_threads = num_threads

        resolved_model = self._prepare_model_path()
        self.session, self.model_loaded_path = self._safe_create_session(resolved_model)
        print(f"[YOLO] Cargando modelo ONNX desde: {self.model_loaded_path}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._supports_batch = self._infer_batch_support()

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
        preds_batch = self._normalize_preds_shape(preds)
        detections = self._postprocess_single(preds_batch[0], scale_x, scale_y, offset_x, offset_y)
        if detections:
            best = max(detections, key=lambda d: d.score)
            print(
                f"[YOLO] detecciones={len(detections)} mejor={best.cls_name} "
                f"score={best.score:.2f} bbox=({best.x},{best.y},{best.w},{best.h})"
            )
        return detections

    def detect_multi_rois(self, frame: np.ndarray, rois: List[ROI]) -> List[YoloDetection]:
        """
        Ejecuta inferencia en batch sobre varias ROIs para maximizar throughput.
        Si la lista está vacía, procesa el frame completo.
        """
        if not rois:
            return self.detect(frame, roi=None)

        blobs: List[np.ndarray] = []
        roi_scales: List[tuple[float, float, int, int]] = []
        for roi in rois:
            if roi.w <= 0 or roi.h <= 0:
                continue
            roi_img = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
            if roi_img.size == 0:
                continue
            blob, scale_x, scale_y = self._preprocess(roi_img)
            blobs.append(blob)
            roi_scales.append((scale_x, scale_y, roi.x, roi.y))

        if not blobs:
            return []

        detections: List[YoloDetection] = []
        if self._supports_batch or len(blobs) == 1:
            batch_blob = np.concatenate(blobs, axis=0)
            preds = self.session.run([self.output_name], {self.input_name: batch_blob})[0]
            preds_batch = self._normalize_preds_shape(preds)
            for idx, (scale_x, scale_y, offset_x, offset_y) in enumerate(roi_scales):
                preds_idx = preds_batch[min(idx, preds_batch.shape[0] - 1)]
                dets_roi = self._postprocess_single(preds_idx, scale_x, scale_y, offset_x, offset_y)
                detections.extend(dets_roi)
        else:
            # El modelo no acepta batch > 1; procesar ROIs una por una.
            for blob, (scale_x, scale_y, offset_x, offset_y) in zip(blobs, roi_scales):
                preds = self.session.run([self.output_name], {self.input_name: blob})[0]
                preds_batch = self._normalize_preds_shape(preds)
                dets_roi = self._postprocess_single(preds_batch[0], scale_x, scale_y, offset_x, offset_y)
                detections.extend(dets_roi)
        if detections:
            best = max(detections, key=lambda d: d.score)
            print(
                f"[YOLO-batch] rois={len(rois)} dets={len(detections)} mejor={best.cls_name} "
                f"score={best.score:.2f}"
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

    def _postprocess_single(
        self, preds: np.ndarray, scale_x: float, scale_y: float, offset_x: int, offset_y: int
    ) -> List[YoloDetection]:
        """
        Convierte la salida de YOLO en detecciones en coordenadas del frame completo.
        Soporta formatos tipo YOLOv8/11 (84, 8400) y tipo Nx(4+cls o 5+cls).
        """
        preds = np.asarray(preds, dtype=np.float32)
        dets: List[YoloDetection] = []

        for row in preds:
            if row.shape[0] < 6:  # necesita al menos bbox + algunas clases
                continue
            # Heurística: si hay muchas columnas (>30) asumimos formato sin obj_conf (YOLOv8/11: 4 + num_classes)
            # si es corto, asumimos obj_conf en row[4] y clases en row[5:].
            if row.shape[0] > 30:
                cls_probs = row[4:]
                obj_conf = 1.0
            else:
                obj_conf = float(row[4])
                cls_probs = row[5:]

            cls_id = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_id])
            score = obj_conf * cls_conf

            if score < self.conf_th:
                continue
            if self.allowed_class_ids is not None and cls_id not in self.allowed_class_ids:
                continue

            cx, cy, w_box, h_box = row[0:4]
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

    def _normalize_preds_shape(self, preds: np.ndarray) -> np.ndarray:
        """
        Adapta salidas comunes de YOLO ONNX:
        - (B, 84, 8400) -> (B, 8400, 84)
        - (B, N, 85) -> (B, N, 85)
        - (N, 85) -> (1, N, 85)
        """
        preds = np.asarray(preds, dtype=np.float32)
        if preds.ndim == 3 and preds.shape[1] in {84, 85}:
            preds = preds.transpose(0, 2, 1)
        elif preds.ndim == 2:
            if preds.shape[0] in {84, 85} and preds.shape[1] > 100:
                preds = preds.transpose(1, 0)
            preds = np.expand_dims(preds, 0)
        return preds

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

    def _pick_providers(self) -> List[str]:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _create_session(self, path: str) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        if self.num_threads and self.num_threads > 0:
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = max(1, self.num_threads // 2)
            print(f"[YOLO] threads intra={opts.intra_op_num_threads} inter={opts.inter_op_num_threads}")
        providers = self._pick_providers()
        return ort.InferenceSession(
            path,
            sess_options=opts,
            providers=providers,
        )

    def _safe_create_session(self, path: str) -> tuple[ort.InferenceSession, str]:
        """
        Intenta crear la sesión; si falla con el modelo cuantizado cae a FP32 para evitar abortar.
        """
        try:
            return self._create_session(path), path
        except Exception as exc:
            if self.use_quantized and path != self.model_path:
                print(f"[YOLO] fallo al cargar modelo cuantizado ({exc}); usando modelo FP32 {self.model_path}")
                self.use_quantized = False
                return self._create_session(self.model_path), self.model_path
            raise

    def _infer_batch_support(self) -> bool:
        """
        Determina si el modelo admite batch > 1 mirando la primera dimensión de entrada.
        """
        try:
            shape = self.session.get_inputs()[0].shape
            if not shape:
                return True
            bdim = shape[0]
            if bdim is None:
                return True
            if isinstance(bdim, str):
                return True
            return int(bdim) != 1  # True si batch libre o >1
        except Exception:
            return True

    def _prepare_model_path(self) -> str:
        if not self.use_quantized:
            return self.model_path
        if self.quantized_model_path and os.path.exists(self.quantized_model_path):
            return self.quantized_model_path
        if quantize_dynamic is None or QuantType is None:
            print("[YOLO] quantize_dynamic no disponible, usando modelo original.")
            return self.model_path
        try:
            dirpath = os.path.dirname(self.quantized_model_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            quantize_dynamic(
                self.model_path,
                self.quantized_model_path,
                weight_type=QuantType.QInt8,
            )
            print(f"[YOLO] modelo cuantizado generado en {self.quantized_model_path}")
            return self.quantized_model_path
        except Exception as exc:  # pragma: no cover - degradar silenciosamente
            print(f"[YOLO] cuantización falló ({exc}), usando modelo original.")
            return self.model_path

    def _default_quantized_path(self, model_path: str) -> str:
        base, ext = os.path.splitext(model_path)
        return f"{base}.int8{ext or '.onnx'}"
