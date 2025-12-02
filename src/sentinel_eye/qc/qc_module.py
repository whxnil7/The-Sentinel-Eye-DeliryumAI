from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class QCMetrics:
    """
    Contenedor de métricas de calidad de imagen en escala 0–100.
    `details` puede incluir valores crudos usados para calcular cada score.
    """

    blur_score: float
    brightness_score: float
    contrast_score: float
    global_score: float
    details: Optional[Dict[str, float]] = None


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

    def evaluate(self, frame: np.ndarray, timestamp: Optional[float] = None) -> QCMetrics:
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

        blur_score = self._estimate_blur(var_lap)
        brightness_score = self._estimate_brightness(mean_brightness, sat_ratio)
        contrast_score = self._estimate_contrast(std_gray)
        global_score = self._aggregate_scores(blur_score, brightness_score, contrast_score)

        details: Dict[str, float] = {
            "var_laplacian": var_lap,
            "mean_brightness": mean_brightness,
            "sat_ratio": sat_ratio,
            "std_gray": std_gray,
        }
        if timestamp is not None:
            details["timestamp"] = float(timestamp)

        return QCMetrics(
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            global_score=global_score,
            details=details,
        )

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

    def _level_from_score(self, score: float) -> str:
        """
        Convierte un score 0–100 en nivel categórico.
        """
        if score >= 80.0:
            return "OK"
        if score >= 50.0:
            return "WARN"
        return "CRITICAL"
