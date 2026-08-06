import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class FloorSurfaceAnalyzer:
    """
    AI Vision Floor Surface & Depth Segmentation Engine V6.0 (Trụ cột 3).
    Analyzes lower region of camera frame to detect floor surface anomalies:
    - Floor cracks, steps, drop-offs
    - Thick carpets / rug boundaries
    - Floor texture changes invisible to 2D 360-degree LiDAR.
    """

    def __init__(self, variance_threshold: float = 450.0):
        self.variance_threshold = variance_threshold

    def analyze_floor_surface(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes lower 40% of the camera frame corresponding to floor plane ahead.
        Returns surface safety status and hazard confidence.
        """
        if frame is None:
            return {"surface_safe": True, "hazard_detected": False, "hazard_type": "NONE"}

        h, w = frame.shape[:2]
        # Crop lower 40% of frame (floor plane region)
        floor_crop = frame[int(h * 0.6):h, int(w * 0.2):int(w * 0.8)]
        gray = cv2.cvtColor(floor_crop, cv2.COLOR_BGR2GRAY)

        # Compute texture variance and edge density using Sobel gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.hypot(sobel_x, sobel_y)

        mean_edge = float(np.mean(edge_magnitude))
        variance = float(np.var(gray))

        hazard_detected = mean_edge > 45.0 or variance > self.variance_threshold
        hazard_type = "SURFACE_ROUGHNESS_OR_CARPET" if hazard_detected else "SMOOTH_FLOOR"

        return {
            "surface_safe": not hazard_detected,
            "hazard_detected": hazard_detected,
            "hazard_type": hazard_type,
            "edge_density": round(mean_edge, 2),
            "texture_variance": round(variance, 2)
        }
