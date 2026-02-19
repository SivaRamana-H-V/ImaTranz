import html
from io import BytesIO
import logging
from typing import List, Dict
from google.cloud import vision
from config.auth import GCPAuth


class GCPServices:
    def __init__(self):
        self.auth = GCPAuth()
        self.vision_client = None
        self.translate_client = None
        self.initialization_error = None

    def initialize(self):
        """Initialize GCP services - returns True/False"""
        success = self.auth.initialize_clients()

        if success:
            self.vision_client = self.auth.vision_client
            self.translate_client = self.auth.translate_client
            return True
        else:
            self.initialization_error = self.auth.initialization_error
            return False

    def vision_ocr_paragraphs(self, pil_img):
        """Extract paragraph-level text using Vision API"""
        if not self.vision_client:
            raise Exception("Vision client not initialized")

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        image = vision.Image(content=buf.getvalue())

        response = self.vision_client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)

        results = []
        annotation = response.full_text_annotation

        for page in annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    words = []
                    for word in para.words:
                        symbols = "".join([s.text for s in word.symbols])
                        words.append(symbols)
                    text = " ".join(words).strip()

                    if not text:
                        continue

                    bbox = para.bounding_box.vertices
                    vertices = [{"x": v.x, "y": v.y} for v in bbox]

                    results.append({
                        "text": text,
                        "bounding_box": vertices
                    })

        return results

    def translate_texts(self, text_list: List[str], target: str = "en") -> List[str]:
        """Translate texts using GCP Translate"""
        if not self.translate_client:
            raise Exception("Translate client not initialized")

        if not text_list:
            return []

        try:
            resp = self.translate_client.translate(
                text_list, target_language=target)
            out = []
            if isinstance(resp, dict):
                out = [html.unescape(resp.get("translatedText", ""))]
            else:
                for it in resp:
                    if isinstance(it, dict):
                        out.append(html.unescape(it.get("translatedText", "")))
                    else:
                        out.append(html.unescape(str(it)))

            if len(out) != len(text_list):
                out = (out + text_list)[:len(text_list)]
            return out
        except Exception as e:
            logging.error(f"GCP Translate error: {e}")
            return text_list
