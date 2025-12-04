from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ROI:
	x: int
	y: int
	w: int
	h: int


# ROIs relativas a todo el frame (no solo a la ROI grande)
# Valores en [0,1] respecto a ancho y alto del frame.
ACTIVITY_ROIS_REL = [
	{
		"name": "road",
		"x_rel": 0.05,
		"y_rel": 0.60,
		"w_rel": 0.90,
		"h_rel": 0.35,
	},
]


def build_activity_rois(frame_width: int, frame_height: int) -> List[ROI]:
	"""Construye las ROIs de actividad a partir de coordenadas relativas."""
	rois: List[ROI] = []
	for cfg in ACTIVITY_ROIS_REL:
		x = int(cfg["x_rel"] * frame_width)
		y = int(cfg["y_rel"] * frame_height)
		w = int(cfg["w_rel"] * frame_width)
		h = int(cfg["h_rel"] * frame_height)
		rois.append(ROI(x=x, y=y, w=w, h=h))
	return rois
