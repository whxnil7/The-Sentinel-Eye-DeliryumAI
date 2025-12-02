from __future__ import annotations

import csv
import os
import time
from typing import Optional

import cv2
import numpy as np

from sentinel_eye.config import ROI as ActivityROI
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
    ) -> None:
        self.video_source: str = video_source
        self.output_path: str = output_path
        self.show_gui: bool = show_gui
        self.qc_log_path: Optional[str] = qc_log_path
        self.yolo_stride: int = max(1, yolo_stride)

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.qc_log_file = None
        self.qc_csv_writer = None
        self._last_yolo_detections: list[YoloDetection] = []

        # TODO: inyectar instancias de módulos cuando los implementemos
        self.qc_module = ImageQualityAssessor()
        self.stability_module = StabilityTracker(roi_fraction=0.5)
        self.motion_module = MotionDetector()
        self.yolo_detector = YoloObjectDetector()

    def _open_capture(self) -> None:
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {self.video_source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # valor por defecto razonable

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = time.time()
            # Aquí más adelante se irán enchufando:
            # frame, qc_metrics = self._run_qc(frame)
            # frame, stability_metrics = self._run_stability(frame)
            # frame, motion_metrics = self._run_motion_detection(frame)

            stability_metrics, roi = self.stability_module.update(frame)
            base_roi = self.stability_module.get_base_roi()
            frame_h, frame_w = frame.shape[:2]
            roi_dyn = self._shift_roi(base_roi or roi, stability_metrics.accum_x, stability_metrics.accum_y, frame_w, frame_h)
            motion_result = self.motion_module.update(frame, roi=roi_dyn)

            qc_metrics, qc_alerts = self.qc_module.evaluate(frame, timestamp=timestamp)
            qc_status = self.qc_module.classify(qc_metrics)
            self._log_qc_metrics(frame_idx, timestamp, qc_metrics, qc_status, qc_alerts, stability_metrics, motion_result)

            yolo_detections: list[YoloDetection] = self._last_yolo_detections
            has_motion = motion_result.metrics.level != "NONE"
            good_qc = qc_metrics.global_score >= 40.0

            dl_blur_bad = qc_alerts.blur_alert
            dl_occ_bad = qc_alerts.occlusion_alert
            dl_light_bad = qc_alerts.light_alert
            very_bad_quality = dl_blur_bad or dl_occ_bad or dl_light_bad

            if has_motion and good_qc and not very_bad_quality:
                if frame_idx % self.yolo_stride == 0:
                    yolo_detections = self.yolo_detector.detect(frame, roi=roi_dyn)
                    self._last_yolo_detections = yolo_detections

            self._draw_basic_overlay(frame, frame_idx)
            self._draw_qc_overlay(frame, qc_metrics, qc_alerts)
            self._draw_stability_overlay(frame, stability_metrics, roi_dyn)
            self._draw_motion_overlay(frame, motion_result)
            self._draw_yolo_overlay(frame, yolo_detections)

            self._write_frame(frame)
            self._display_frame(frame, frame_idx)

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

        self._release()

    def _draw_basic_overlay(self, frame: np.ndarray, frame_idx: int) -> None:
        """
        Por ahora solo dibuja el número de frame.
        Más adelante agregaremos QC_Score, gráficos de vibración y bounding boxes.
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

    def _draw_motion_overlay(self, frame: np.ndarray, motion_result: MotionResult) -> None:
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

        for box in motion_result.boxes:
            cv2.rectangle(
                frame,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                color,
                2,
            )

        area_pct = motion_result.metrics.total_moving_area_ratio * 100.0
        text = f"MOT: {level}   regions: {motion_result.metrics.num_regions}   area: {area_pct:.1f} %"
        cv2.putText(
            frame,
            text,
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        color_roi = (0, 255, 0)
        if motion_result.activity_rois:
            for roi in motion_result.activity_rois:
                x1, y1 = roi.x, roi.y
                x2, y2 = roi.x + roi.w, roi.y + roi.h
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_roi, 2)
        if hasattr(motion_result, "roi") and motion_result.activity_rois == []:
            # legacy path; no extra drawing needed
            pass

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
                "global_score",
                "var_laplacian",
                "mean_brightness",
                "sat_ratio",
                "std_gray",
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
                qc.global_score,
                qc.var_laplacian,
                qc.mean_brightness,
                qc.sat_ratio,
                qc.std_gray,
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

    def _draw_qc_overlay(self, frame: np.ndarray, qc: QCMetrics, alerts: QCAlerts) -> None:
        """
        Dibuja los scores de QC en la esquina superior izquierda.
        El color depende del score global: verde/amarillo/rojo.
        """
        color = self._qc_color_from_level(self.qc_module._level_from_score(qc.global_score))
        # Panel de fondo semitransparente para legibilidad
        panel_x, panel_y, panel_w, panel_h = 5, 35, 560, 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        y = panel_y + 22
        self._put_text(frame, f"QC: {qc.global_score:.1f}", (10, y), color, font_scale=0.85, thickness=2)
        self._put_text(
            frame,
            f"Blur: {qc.blur_score:.1f}  Bright: {qc.brightness_score:.1f}  Contr: {qc.contrast_score:.1f}",
            (10, y + 22),
            color,
            font_scale=0.75,
            thickness=2,
        )
        y = y + 22 + 6
        cat_color = (0, 255, 255)
        if alerts.blur_alert or alerts.occlusion_alert or alerts.light_alert:
            cat_color = (0, 0, 255)
        text = f"Blur: {alerts.blur_level}   Occ: {alerts.occlusion_level}   Light/Glare: {alerts.light_level}"
        self._put_text(frame, text, (10, y + 20), cat_color, font_scale=0.75, thickness=2)
        if qc.details and "dl_p_blur_bad" in qc.details:
            text = (
                f"DL-QC blur={qc.details['dl_p_blur_bad']:.2f} "
                f"occ={qc.details['dl_p_occlusion_bad']:.2f} "
                f"light={qc.details['dl_p_light_bad']:.2f}"
            )
            self._put_text(frame, text, (10, y + 42), (200, 200, 0), font_scale=0.7, thickness=2)

    def _draw_stability_overlay(self, frame: np.ndarray, stab: StabilityMetrics, roi: ROI) -> None:
        """
        Dibuja ROI compensada y métricas de estabilidad.
        """
        color = self._stab_color(stab.level)
        text = f"STAB: {stab.level}  dx: {stab.dx:.1f}  dy: {stab.dy:.1f}  |mag|: {stab.mag:.1f}"
        self._put_text(frame, text, (10, 140), color, font_scale=0.7, thickness=2)

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

    def _stab_color(self, level: str) -> tuple[int, int, int]:
        if level == "STABLE":
            return (0, 200, 0)  # verde
        if level == "WOBBLE":
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

    def _release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        self._close_qc_log()
        if self.show_gui:
            cv2.destroyAllWindows()
