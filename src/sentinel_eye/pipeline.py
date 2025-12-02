from __future__ import annotations

import csv
import os
import time
from typing import Optional

import cv2
import numpy as np

from sentinel_eye.motion.motion_module import MotionDetector, MotionResult
from sentinel_eye.qc.qc_module import ImageQualityAssessor, QCStatus, QCMetrics
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
    ) -> None:
        self.video_source: str = video_source
        self.output_path: str = output_path
        self.show_gui: bool = show_gui
        self.qc_log_path: Optional[str] = qc_log_path

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.qc_log_file = None
        self.qc_csv_writer = None

        # TODO: inyectar instancias de módulos cuando los implementemos
        self.qc_module = ImageQualityAssessor()
        self.stability_module = StabilityTracker(roi_fraction=0.5)
        self.motion_module = MotionDetector()

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
            motion_result = self.motion_module.update(frame, roi=roi)

            qc_metrics = self.qc_module.evaluate(frame, timestamp=timestamp)
            qc_status = self.qc_module.classify(qc_metrics)
            self._log_qc_metrics(frame_idx, timestamp, qc_metrics, qc_status, stability_metrics, motion_result)

            self._draw_basic_overlay(frame, frame_idx)
            self._draw_qc_overlay(frame, qc_metrics, qc_status)
            self._draw_stability_overlay(frame, stability_metrics, roi)
            self._draw_motion_overlay(frame, motion_result)

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
        stability: StabilityMetrics,
        motion: MotionResult,
    ) -> None:
        if self.qc_csv_writer is None:
            return
        details = qc.details or {}
        self.qc_csv_writer.writerow(
            [
                frame_idx,
                timestamp,
                qc.blur_score,
                qc.brightness_score,
                qc.contrast_score,
                qc.global_score,
                details.get("var_laplacian", 0.0),
                details.get("mean_brightness", 0.0),
                details.get("sat_ratio", 0.0),
                details.get("std_gray", 0.0),
                status.global_level,
                status.blur,
                status.brightness,
                status.contrast,
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

    def _draw_qc_overlay(self, frame: np.ndarray, qc: QCMetrics, status: QCStatus) -> None:
        """
        Dibuja los scores de QC en la esquina superior izquierda.
        El color depende del score global: verde/amarillo/rojo.
        """
        color = self._qc_color_from_level(status.global_level)
        cv2.putText(
            frame,
            f"QC: {qc.global_score:.1f} [{status.global_level}]",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Blur: {qc.blur_score:.1f} ({status.blur})  "
                f"Bright: {qc.brightness_score:.1f} ({status.brightness})  "
                f"Contr: {qc.contrast_score:.1f} ({status.contrast})"
            ),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_stability_overlay(self, frame: np.ndarray, stab: StabilityMetrics, roi: ROI) -> None:
        """
        Dibuja ROI compensada y métricas de estabilidad.
        """
        color = self._stab_color(stab.level)
        cv2.rectangle(
            frame,
            (roi.x, roi.y),
            (roi.x + roi.w, roi.y + roi.h),
            color,
            2,
        )
        text = f"STAB: {stab.level}  dx: {stab.dx:.1f}  dy: {stab.dy:.1f}  |mag|: {stab.mag:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _qc_color_from_level(self, level: str) -> tuple[int, int, int]:
        if level == "OK":
            return (0, 200, 0)  # verde
        if level == "WARN":
            return (0, 200, 200)  # amarillo
        return (0, 0, 255)  # rojo

    def _stab_color(self, level: str) -> tuple[int, int, int]:
        if level == "STABLE":
            return (0, 200, 0)  # verde
        if level == "WOBBLE":
            return (0, 200, 200)  # amarillo
        return (0, 0, 255)  # rojo

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
