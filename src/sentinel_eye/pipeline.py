from __future__ import annotations

import csv
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

from sentinel_eye.config import ROI as ActivityROI, build_activity_rois
from sentinel_eye.motion.motion_module import MotionDetector, MotionResult
from sentinel_eye.motion.yolo_detector import YoloDetection, YoloObjectDetector
from sentinel_eye.qc.qc_module import ImageQualityAssessor, QCAlerts, QCStatus, QCMetrics
from sentinel_eye.stability.stability_module import ROI, StabilityMetrics, StabilityTracker


class VideoPipeline:
    """
    Pipeline principal que:
    - Lee frames desde un video.
    - Llama a los módulos de QC, estabilidad y detección de movimiento (más adelante).
    - Escribe un video de salida con overlays.
    - Opcionalmente muestra una ventana (para desarrollo local).
    """

    def __init__(
        self,
        video_source: str,
        output_path: str,
        show_gui: bool = True,
        qc_log_path: Optional[str] = "data/output/qc_metrics.csv",
        yolo_stride: int = 3,
        yolo_model_path: str = "models/yolo11n.onnx",
        yolo_conf: float = 0.28,
        yolo_iou: float = 0.5,
        yolo_allow_all_classes: bool = False,
        yolo_use_quantized: bool = True,
        yolo_quantized_path: Optional[str] = None,
        yolo_num_threads: Optional[int] = None,
        preview_real_time: bool = False,
    ) -> None:
        self.video_source: str = video_source
        self.output_path: str = output_path
        self.show_gui: bool = show_gui
        self.qc_log_path: Optional[str] = qc_log_path
        self.yolo_stride: int = max(1, yolo_stride)
        self.yolo_model_path = yolo_model_path
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.yolo_allow_all_classes = yolo_allow_all_classes
        self.yolo_use_quantized = yolo_use_quantized
        self.yolo_quantized_path = yolo_quantized_path
        self.yolo_num_threads = yolo_num_threads
        self._stab_history: list[tuple[float, float]] = []
        self._stab_history_max_len: int = 160
        self.activity_rois: Optional[list[ActivityROI]] = None
        self._activity_rois_base: Optional[list[ActivityROI]] = None
        self._yolo_calls: int = 0
        self._yolo_kept: int = 0
        self._yolo_raw: int = 0
        self.preview_real_time: bool = preview_real_time
        self.source_fps: float = 25.0
        self._progress_total: int = 0
        self._progress_last = -1

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.qc_log_file = None
        self.qc_csv_writer = None
        self._last_yolo_detections: list[YoloDetection] = []

        # TODO: inyectar instancias de módulos cuando los implementemos
        self.qc_module = ImageQualityAssessor()
        self.stability_module = StabilityTracker(roi_fraction=0.5)
        self.motion_module = MotionDetector()
        allowed_ids = None if yolo_allow_all_classes else {2, 3, 5, 7}
        self.yolo_detector = YoloObjectDetector(
            model_path=self.yolo_model_path,
            conf_th=self.yolo_conf,
            iou_th=self.yolo_iou,
            allowed_class_ids=allowed_ids,
            use_quantized=self.yolo_use_quantized,
            quantized_model_path=self.yolo_quantized_path,
            num_threads=self.yolo_num_threads,
        )

    def _open_capture(self) -> None:
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {self.video_source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # valor por defecto razonable
        self.source_fps = fps

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._progress_total = total_frames if total_frames > 0 else 0

        self._open_writer(width, height, fps)

    def _open_writer(self, width: int, height: int, fps: float) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"No se pudo crear el video de salida: {self.output_path}")

    def run(self) -> None:
        self._open_capture()
        self._init_qc_log()
        assert self.cap is not None

        frame_idx = 0
        t0 = time.time()
        processed_frames = 0
        self._init_progress()

        while True:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.activity_rois is None:
                h, w = frame.shape[:2]
                self.activity_rois = build_activity_rois(w, h)
                self._activity_rois_base = [ActivityROI(r.x, r.y, r.w, r.h) for r in self.activity_rois]

            timestamp = time.time()
            # Aquí más adelante se irán enchufando:
            # frame, qc_metrics = self._run_qc(frame)
            # frame, stability_metrics = self._run_stability(frame)
            # frame, motion_metrics = self._run_motion_detection(frame)

            stability_metrics, roi = self.stability_module.update(frame)
            base_roi = self.stability_module.get_base_roi()
            frame_h, frame_w = frame.shape[:2]

            roi_dyn = self._shift_roi(base_roi or roi, stability_metrics.accum_x, stability_metrics.accum_y, frame_w, frame_h)
            self._update_stab_history(stability_metrics)

            # Usar únicamente la ROI compensada de estabilidad como ROI de actividad (sin franja “Carretera”).
            current_activity = [ActivityROI(roi_dyn.x, roi_dyn.y, roi_dyn.w, roi_dyn.h)]
            self.activity_rois = current_activity
            self.motion_module.activity_rois = current_activity
            motion_result = self.motion_module.update(frame, roi=roi_dyn)

            qc_metrics, qc_alerts = self.qc_module.evaluate(frame, timestamp=timestamp)
            qc_status = self.qc_module.classify(qc_metrics)
            self._log_qc_metrics(frame_idx, timestamp, qc_metrics, qc_status, qc_alerts, stability_metrics, motion_result)

            yolo_detections: list[YoloDetection] = self._last_yolo_detections
            # Para YOLO procesamos frame completo y sin filtrar por ROIs de actividad.
            detect_roi = None
            detect_rois_list: list[ROI] = []
            saved_activity_rois = self.activity_rois
            self.activity_rois = []
            if frame_idx % self.yolo_stride == 0:
                raw_dets = self.yolo_detector.detect_multi_rois(frame, rois=detect_rois_list)
                yolo_detections = self._postprocess_detections(raw_dets, detect_roi)
                self._last_yolo_detections = yolo_detections
                self._yolo_calls += 1
                self._yolo_raw += len(raw_dets)
                self._yolo_kept += len(yolo_detections)
            else:
                yolo_detections = self._postprocess_detections(self._last_yolo_detections, detect_roi)
            self.activity_rois = saved_activity_rois

            # Asegura que las detecciones finales queden confinadas a la unión de ROIs activas actuales.
            detect_roi = self._activity_union_roi()
            yolo_detections = self._postprocess_detections(yolo_detections, detect_roi)

            self._draw_info_panel(frame, frame_idx, qc_metrics, qc_alerts, stability_metrics, motion_result)
            self._draw_stability_rois(frame, base_roi, roi_dyn)
            self._draw_motion_overlay(frame, motion_result, base_rois=self._activity_rois_base)
            self._draw_stability_plot(frame)
            self._draw_yolo_overlay(frame, yolo_detections)

            self._write_frame(frame)
            self._display_frame(frame, frame_idx)
            self._throttle_preview(frame_start)
            self._update_progress(processed_frames + 1)

            processed_frames += 1
            frame_idx += 1

            # Salir con la tecla 'q' (solo si hay GUI)
            if self.show_gui:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        t1 = time.time()
        elapsed = max(t1 - t0, 1e-6)
        fps = processed_frames / elapsed
        print(f"[VideoPipeline] Frames procesados: {processed_frames} | FPS ~ {fps:.2f}")
        print(f"[VideoPipeline] Video de salida: {self.output_path}")
        print(
            f"[YOLO] Llamadas efectivas: {self._yolo_calls} | Detecciones crudas totales: {self._yolo_raw} "
            f"| Detecciones mantenidas (suma): {self._yolo_kept}"
        )
        self._finish_progress()

        self._release()

    def _draw_motion_overlay(
        self,
        frame: np.ndarray,
        motion_result: MotionResult,
        base_rois: Optional[list[ActivityROI]] = None,
    ) -> None:
        """
        Dibuja bounding boxes de movimiento y resume nivel/área.
        """
        level = motion_result.metrics.level
        if level == "HIGH":
            color = (0, 0, 255)  # rojo
        elif level == "LOW":
            color = (0, 200, 200)  # amarillo
        else:
            color = (0, 200, 0)  # verde

        # No dibujar bounding boxes de movimiento para evitar el rectángulo fino.

        area_pct = motion_result.metrics.total_moving_area_ratio * 100.0
        color_roi = (0, 255, 0)
        if base_rois:
            for base in base_rois:
                x1b, y1b = base.x, base.y
                x2b, y2b = base.x + base.w, base.y + base.h
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (80, 80, 200), 1)

        # Oculta dibujo/etiqueta de las ROIs de actividad (solo usamos la ROI central de estabilidad).

    def _draw_yolo_overlay(self, frame: np.ndarray, detections: list[YoloDetection]) -> None:
        """
        Dibuja detecciones YOLO con cajas y etiquetas.
        """
        for det in detections:
            color = (0, 255, 255)  # cian/amarillo brillante para distinguir de motion
            x1, y1, w, h = det.x, det.y, det.w, det.h
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.cls_name} {det.score:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _write_frame(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def _display_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        if not self.show_gui:
            return
        cv2.imshow("The Sentinel Eye - preview", frame)

    def _throttle_preview(self, frame_start: float) -> None:
        """Si preview_real_time está activo, espera para igualar el fps de origen."""
        if not (self.show_gui and self.preview_real_time):
            return
        if self.source_fps <= 0:
            return
        target_dt = 1.0 / self.source_fps
        elapsed = time.time() - frame_start
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _init_qc_log(self) -> None:
        """
        Prepara el archivo CSV para loguear métricas de QC, si está configurado.
        """
        if not self.qc_log_path:
            return

        dirpath = os.path.dirname(self.qc_log_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.qc_log_file = open(self.qc_log_path, "w", newline="")
        self.qc_csv_writer = csv.writer(self.qc_log_file)
        self.qc_csv_writer.writerow(
            [
                "frame_idx",
                "timestamp",
                "blur_score",
                "brightness_score",
                "contrast_score",
                "occlusion_score",
                "global_score",
                "var_laplacian",
                "mean_brightness",
                "sat_ratio",
                "std_gray",
                "edge_density",
                "blocked_ratio",
                "global_level",
                "blur_level",
                "brightness_level",
                "contrast_level",
                "qc_blur_level",
                "qc_occlusion_level",
                "qc_light_level",
                "qc_blur_alert",
                "qc_occlusion_alert",
                "qc_light_alert",
                "stab_dx",
                "stab_dy",
                "stab_mag",
                "stab_vibration_rms",
                "stab_level",
                "stab_accum_x",
                "stab_accum_y",
                "mot_num_regions",
                "mot_area_ratio",
                "mot_level",
            ]
        )

    def _log_qc_metrics(
        self,
        frame_idx: int,
        timestamp: float,
        qc: QCMetrics,
        status: QCStatus,
        alerts: QCAlerts,
        stability: StabilityMetrics,
        motion: MotionResult,
    ) -> None:
        if self.qc_csv_writer is None:
            return
        self.qc_csv_writer.writerow(
            [
                frame_idx,
                timestamp,
                qc.blur_score,
                qc.brightness_score,
                qc.contrast_score,
                qc.occlusion_score,
                qc.global_score,
                qc.var_laplacian,
                qc.mean_brightness,
                qc.sat_ratio,
                qc.std_gray,
                qc.edge_density,
                qc.blocked_ratio,
                status.global_level,
                status.blur,
                status.brightness,
                status.contrast,
                alerts.blur_level,
                alerts.occlusion_level,
                alerts.light_level,
                int(alerts.blur_alert),
                int(alerts.occlusion_alert),
                int(alerts.light_alert),
                stability.dx,
                stability.dy,
                stability.mag,
                stability.vibration_rms,
                stability.level,
                stability.accum_x,
                stability.accum_y,
                motion.metrics.num_regions,
                motion.metrics.total_moving_area_ratio,
                motion.metrics.level,
            ]
        )
        if self.qc_log_file:
            self.qc_log_file.flush()

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        frame_idx: int,
        qc: QCMetrics,
        alerts: QCAlerts,
        stab: StabilityMetrics,
        motion: MotionResult,
    ) -> None:
        """Panel compacto en la esquina superior izquierda con QC, estabilidad y motion."""
        qc_color = self._qc_color_from_level(self.qc_module._level_from_score(qc.global_score))
        cat_color = (0, 255, 255)
        if alerts.blur_alert or alerts.occlusion_alert or alerts.light_alert:
            cat_color = (0, 0, 255)
        stab_color = self._stab_color(stab.level)
        mot_color = (0, 200, 0)
        if motion.metrics.level == "HIGH":
            mot_color = (0, 0, 255)
        elif motion.metrics.level == "LOW":
            mot_color = (0, 200, 200)

        panel_x, panel_y, panel_w, panel_h = 5, 5, 580, 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        y = panel_y + 22
        self._put_text(frame, f"Frame: {frame_idx}", (10, y), qc_color, font_scale=0.85, thickness=2)
        y += 22
        self._put_text(
            frame,
            f"QC {qc.global_score:.1f}  |  Blur {qc.blur_score:.1f}  Bright {qc.brightness_score:.1f}  Contr {qc.contrast_score:.1f}",
            (10, y),
            qc_color,
            font_scale=0.75,
            thickness=2,
        )
        y += 22
        self._put_text(
            frame,
            f"Levels: Blur {alerts.blur_level}  Occ {alerts.occlusion_level}  Light {alerts.light_level}",
            (10, y),
            cat_color,
            font_scale=0.75,
            thickness=2,
        )
        if qc.details and "dl_p_blur_bad" in qc.details:
            y += 18
            self._put_text(
                frame,
                f"DL-QC  blur={qc.details['dl_p_blur_bad']:.2f}  occ={qc.details['dl_p_occlusion_bad']:.2f}  light={qc.details['dl_p_light_bad']:.2f}",
                (10, y),
                (200, 200, 0),
                font_scale=0.65,
                thickness=2,
            )
        y += 22
        self._put_text(
            frame,
            f"Stab {stab.level}  dx {stab.dx:.1f}  dy {stab.dy:.1f}  |mag| {stab.mag:.1f}  vib {stab.vibration_rms:.2f}  tracks {stab.num_tracked}",
            (10, y),
            stab_color,
            font_scale=0.7,
            thickness=2,
        )
        y += 20
        area_pct = motion.metrics.total_moving_area_ratio * 100.0
        self._put_text(
            frame,
            f"Motion {motion.metrics.level}  regions {motion.metrics.num_regions}  area {area_pct:.1f} %",
            (10, y),
            mot_color,
            font_scale=0.7,
            thickness=2,
        )

    def _qc_color_from_level(self, level: str) -> tuple[int, int, int]:
        if level == "OK":
            return (0, 200, 0)  # verde
        if level == "WARN":
            return (0, 200, 200)  # amarillo
        return (0, 0, 255)  # rojo

    def _shift_roi(self, base_roi: Optional[ROI], dx: float, dy: float, frame_w: int, frame_h: int) -> ActivityROI:
        if base_roi is None:
            return ActivityROI(0, 0, frame_w, frame_h)
        shift_x = int(round(dx))
        shift_y = int(round(dy))
        new_x = base_roi.x - shift_x
        new_y = base_roi.y - shift_y
        new_x = max(0, min(new_x, frame_w - base_roi.w))
        new_y = max(0, min(new_y, frame_h - base_roi.h))
        return ActivityROI(new_x, new_y, base_roi.w, base_roi.h)

    def _shift_activity_rois(self, rois: list[ActivityROI], dx: float, dy: float, frame_w: int, frame_h: int) -> list[ActivityROI]:
        """Desplaza las ROIs de actividad compensando el movimiento estimado de la cámara."""
        shift_x = int(round(dx))
        shift_y = int(round(dy))
        shifted: list[ActivityROI] = []
        for r in rois:
            new_x = max(0, min(r.x - shift_x, frame_w - r.w))
            new_y = max(0, min(r.y - shift_y, frame_h - r.h))
            shifted.append(ActivityROI(new_x, new_y, r.w, r.h))
        return shifted

    def _activity_union_roi(self) -> Optional[ActivityROI]:
        """Devuelve una ROI que cubre las dos ROIs de actividad inferiores."""
        if not self.activity_rois:
            return None
        min_x = min(r.x for r in self.activity_rois)
        min_y = min(r.y for r in self.activity_rois)
        max_x = max(r.x + r.w for r in self.activity_rois)
        max_y = max(r.y + r.h for r in self.activity_rois)
        return ActivityROI(min_x, min_y, max_x - min_x, max_y - min_y)

    def _filter_detections_by_activity(self, detections: list[YoloDetection]) -> list[YoloDetection]:
        """Mantiene solo detecciones cuyo centro caiga dentro de alguna ROI de actividad."""
        if not self.activity_rois:
            return detections
        filtered: list[YoloDetection] = []
        for det in detections:
            cx = det.x + det.w / 2.0
            cy = det.y + det.h / 2.0
            inside = False
            for roi in self.activity_rois:
                if roi.x <= cx <= roi.x + roi.w and roi.y <= cy <= roi.y + roi.h:
                    inside = True
                    break
            if inside:
                filtered.append(det)
        return filtered

    def _postprocess_detections(self, detections: list[YoloDetection], roi: Optional[ActivityROI]) -> list[YoloDetection]:
        """Filtra por ROIs de actividad y recorta a la ROI unión."""
        filtered = self._filter_detections_by_activity(detections)
        clipped = self._clip_detections_to_roi_union(filtered, roi)
        return clipped

    def _clip_detections_to_roi_union(
        self, detections: list[YoloDetection], roi: Optional[ActivityROI]
    ) -> list[YoloDetection]:
        """Recorta las cajas para que no sobresalgan de la ROI unión y descarta las que no intersectan."""
        if roi is None:
            return detections
        clipped: list[YoloDetection] = []
        for det in detections:
            x1 = max(det.x, roi.x)
            y1 = max(det.y, roi.y)
            x2 = min(det.x + det.w, roi.x + roi.w)
            y2 = min(det.y + det.h, roi.y + roi.h)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            clipped.append(
                YoloDetection(
                    x=int(x1),
                    y=int(y1),
                    w=int(w),
                    h=int(h),
                    cls_id=det.cls_id,
                    cls_name=det.cls_name,
                    score=det.score,
                )
            )
        return clipped

    def _update_stab_history(self, stab: StabilityMetrics) -> None:
        """Mantiene una ventana deslizante de desplazamientos acumulados para graficar vibración/deriva."""
        self._stab_history.append((stab.accum_x, stab.accum_y))
        if len(self._stab_history) > self._stab_history_max_len:
            self._stab_history.pop(0)

    def _draw_stability_rois(self, frame: np.ndarray, base_roi: Optional[ROI], compensated_roi: Optional[ActivityROI]) -> None:
        """
        Dibuja la ROI base (referencia) y la ROI compensada que sigue la deriva.
        """
        if base_roi is not None:
            cv2.rectangle(
                frame,
                (base_roi.x, base_roi.y),
                (base_roi.x + base_roi.w, base_roi.y + base_roi.h),
                (120, 120, 255),
                1,
            )
        if compensated_roi is not None:
            cv2.rectangle(
                frame,
                (compensated_roi.x, compensated_roi.y),
                (compensated_roi.x + compensated_roi.w, compensated_roi.y + compensated_roi.h),
                (0, 180, 255),
                2,
            )

    def _draw_stability_plot(self, frame: np.ndarray) -> None:
        """Grafica desplazamiento acumulado X/Y en un panel pequeño para ver vibración/deriva."""
        if not self._stab_history:
            return
        _, w = frame.shape[:2]
        panel_w, panel_h = 220, 120
        panel_x = max(5, w - panel_w - 10)
        panel_y = 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        center_y = panel_y + panel_h // 2
        max_points = min(len(self._stab_history), panel_w - 20)
        recent = self._stab_history[-max_points:]

        max_abs = max([max(abs(x), abs(y)) for x, y in recent] + [1.0])
        scale = (panel_h * 0.45) / max_abs

        pts_x = []
        pts_y = []
        for i, (sx, sy) in enumerate(recent):
            px = panel_x + 10 + int((i / max(1, max_points - 1)) * (panel_w - 20))
            py_x = int(center_y - sx * scale)
            py_y = int(center_y - sy * scale)
            pts_x.append((px, py_x))
            pts_y.append((px, py_y))

        cv2.line(frame, (panel_x + 5, center_y), (panel_x + panel_w - 5, center_y), (80, 80, 80), 1)
        cv2.line(frame, (panel_x + 10, panel_y + 8), (panel_x + 10, panel_y + panel_h - 8), (80, 80, 80), 1)
        if len(pts_x) >= 2:
            cv2.polylines(frame, [np.array(pts_x, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
        if len(pts_y) >= 2:
            cv2.polylines(frame, [np.array(pts_y, dtype=np.int32)], isClosed=False, color=(0, 200, 255), thickness=2)

        self._put_text(frame, "drift X", (panel_x + 14, panel_y + 18), (0, 255, 0), font_scale=0.55, thickness=1)
        self._put_text(frame, "drift Y", (panel_x + 14, panel_y + 36), (0, 200, 255), font_scale=0.55, thickness=1)
        self._put_text(frame, "hist", (panel_x + panel_w - 48, panel_y + panel_h - 10), (200, 200, 200), font_scale=0.5, thickness=1)

    def _stab_color(self, level: str) -> tuple[int, int, int]:
        if level == "STABLE":
            return (0, 200, 0)  # verde
        if level in {"WOBBLE", "VIBRATION"}:
            return (0, 200, 200)  # amarillo
        return (0, 0, 255)  # rojo

    def _put_text(
        self,
        frame: np.ndarray,
        text: str,
        org: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
        """Dibuja texto con sombra para mejor legibilidad."""
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _close_qc_log(self) -> None:
        if self.qc_log_file is not None:
            self.qc_log_file.close()
            self.qc_log_file = None
            self.qc_csv_writer = None

    def _init_progress(self) -> None:
        if self._progress_total <= 0:
            return
        sys.stdout.write("[Progress] 0%\r")
        sys.stdout.flush()

    def _update_progress(self, processed: int) -> None:
        if self._progress_total <= 0:
            return
        pct = int((processed / float(self._progress_total)) * 100)
        if pct != self._progress_last:
            sys.stdout.write(f"[Progress] {pct}%\r")
            sys.stdout.flush()
            self._progress_last = pct

    def _finish_progress(self) -> None:
        if self._progress_total <= 0:
            return
        sys.stdout.write("[Progress] 100%\n")
        sys.stdout.flush()

    def _release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        self._close_qc_log()
        if self.show_gui:
            cv2.destroyAllWindows()
