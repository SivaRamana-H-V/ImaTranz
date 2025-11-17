from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

@dataclass
class TextAnnotation:
    text: str
    bounding_box: List[Dict[str, float]]
    translated_text: Optional[str] = None
    confidence: Optional[float] = None

@dataclass
class ProcessedImage:
    original_url: str
    original_image: any  # PIL Image
    cleaned_image: any   # PIL Image
    final_image: any     # PIL Image
    annotations: List[TextAnnotation]
    metadata: Dict