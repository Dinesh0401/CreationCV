"""Vision module for ReelMotion AI — Phase 3.

Provides:
- Pose estimation (MediaPipe PoseLandmarker / OpenCV skeleton fallback)
- Face tracking (478 or 68 landmarks, Eye Aspect Ratio EAR, Mouth Aspect Ratio MAR)
- Hand tracking (21 landmarks for left and right hands)
- Person segmentation (binary and alpha masks, background isolation)
"""

from backend.vision.face_tracking import (
    BlinkMetrics,
    FaceLandmark,
    FaceTracker,
    FaceTrackingResult,
    MouthMetrics,
)
from backend.vision.hand_tracking import (
    HAND_CONNECTIONS,
    HandLandmark,
    HandLandmarkIndex,
    HandTracker,
    HandTrackingResult,
    SingleHandResult,
)
from backend.vision.pose import (
    POSE_CONNECTIONS,
    NormalizedLandmark,
    PoseEstimator,
    PoseLandmarkIndex,
    PoseResult,
)
from backend.vision.segmentation import (
    PersonSegmenter,
    SegmentationMetadata,
    SegmentationResult,
)

__all__ = [
    "HAND_CONNECTIONS",
    "POSE_CONNECTIONS",
    "BlinkMetrics",
    "FaceLandmark",
    "FaceTracker",
    "FaceTrackingResult",
    "HandLandmark",
    "HandLandmarkIndex",
    "HandTracker",
    "HandTrackingResult",
    "MouthMetrics",
    "NormalizedLandmark",
    "PersonSegmenter",
    "PoseEstimator",
    "PoseLandmarkIndex",
    "PoseResult",
    "SegmentationMetadata",
    "SegmentationResult",
    "SingleHandResult",
]
