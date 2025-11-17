import textwrap
import html
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
from .image_utils import sample_background_color, choose_text_color
from .geometry_utils import get_text_angle, poly_to_rect_coords, normalize_polygon

class TextOverlay:
    def __init__(self, font_path: str, min_font_size: int = 12):
        self.font_path = font_path
        self.min_font_size = min_font_size

    def pick_font_for_size(self, w_limit, h_limit, text, vertical=False):
        """Select appropriate font size for given dimensions"""
        for fs in range(72, self.min_font_size - 1, -1):
            f = ImageFont.truetype(self.font_path, fs)
            tmp = Image.new("RGBA", (2000, 800), (0, 0, 0, 0))
            d = ImageDraw.Draw(tmp)

            if vertical:
                bbox = d.textbbox((0, 0), text, font=f)
            else:
                wrapped = "\n".join(textwrap.wrap(
                    text, width=max(1, int(w_limit / max(1, (fs * 0.6)))),
                    break_on_hyphens=False, break_long_words=False
                ))
                bbox = d.multiline_textbbox((0, 0), wrapped, font=f)

            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if tw <= w_limit * 1.05 and th <= h_limit * 1.05:
                return f, (tw, th), wrapped if not vertical else text

        f = ImageFont.truetype(self.font_path, self.min_font_size)
        return f, (w_limit, h_limit), text

    def draw_rotated_text_block(self, base_img, text, font, angle, box_rect):
        """Draw rotated text block"""
        x1, y1, x2, y2 = box_rect
        box_w = x2 - x1
        box_h = y2 - y1

        temp = Image.new("RGBA", (2000, 500), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp)
        draw.text((0, 0), text, font=font, fill="black")

        bbox = temp.getbbox()
        text_img = temp.crop(bbox)
        rotated = text_img.rotate(angle, expand=True)

        rx = x1 + (box_w - rotated.width) / 2
        ry = y1 + (box_h - rotated.height) / 2

        base_img.alpha_composite(rotated, dest=(int(rx), int(ry)))
        return base_img

    def overlay_translated_text(self, img_pil: Image.Image, annotations: List[Dict]) -> Image.Image:
        """Overlay translated text on image"""
        out = img_pil.convert("RGBA")
        w_img, h_img = out.size

        for ann in annotations:
            raw_poly = ann.get("bounding_box") or ann.get("bbox") or ann.get("poly") or ann.get("polygon")
            if not raw_poly:
                continue
            poly = normalize_polygon(raw_poly)
            if not poly:
                continue

            x1, y1, x2, y2 = poly_to_rect_coords(poly)
            box_w = max(3, x2 - x1)
            box_h = max(3, y2 - y1)

            new_text = (ann.get("new_text") or ann.get("translated_text") or ann.get("text") or "").strip()
            if not new_text:
                continue

            aspect = float(box_h) / max(1.0, box_w)
            want_vertical = aspect >= 2.5

            # Vertical text handling
            if want_vertical:
                font, (tw, th), _ = self.pick_font_for_size(box_w, box_h, new_text, vertical=True)
                angle = -90 if get_text_angle(poly) < -30 else 270
                out = self.draw_rotated_text_block(out, new_text, font, angle, (x1, y1, x2, y2))
                continue

            # Horizontal text handling
            font, (tw, th), wrapped = self.pick_font_for_size(box_w, box_h, new_text, vertical=False)

            needed_w = tw + int(font.size * 1.8)
            needed_h = th + int(font.size * 1.6)

            expand_x = max(0, int((needed_w - box_w) / 2))
            expand_y = max(0, int((needed_h - box_h) / 2))

            expand_x = min(expand_x, int(w_img * 0.30))
            expand_y = min(expand_y, int(h_img * 0.20))

            px1 = max(0, x1 - expand_x)
            py1 = max(0, y1 - expand_y)
            px2 = min(w_img, x2 + expand_x)
            py2 = min(h_img, y2 + expand_y)

            place_w = px2 - px1
            place_h = py2 - py1

            avg_rgb = sample_background_color(img_pil, (px1, py1, px2, py2))
            text_color = choose_text_color(avg_rgb)
            stroke = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)

            tmp = Image.new("RGBA", (place_w, place_h), (255, 255, 255, 0))
            d = ImageDraw.Draw(tmp)

            padding = max(4, int(font.size * 0.3))
            spacing = max(1, int(font.size * 0.12))

            bb = d.multiline_textbbox((padding, padding), wrapped, font=font, spacing=spacing)
            bx0, by0, bx1, by1 = bb
            d.rectangle([bx0-3, by0-3, bx1+3, by1+3], fill=(avg_rgb[0], avg_rgb[1], avg_rgb[2], 210))

            d.multiline_text(
                (padding, padding), wrapped,
                font=font, fill=text_color,
                spacing=spacing,
                stroke_width=max(1, int(font.size * 0.06)),
                stroke_fill=stroke
            )

            angle = get_text_angle(poly)
            if abs(angle) > 15:
                angle = 0

            rotated = tmp.rotate(-angle, expand=True)
            rx = px1 + (place_w - rotated.width) / 2
            ry = py1 + (place_h - rotated.height) / 2

            out.alpha_composite(rotated, dest=(int(rx), int(ry)))

        return out.convert("RGB")