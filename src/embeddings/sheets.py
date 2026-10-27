"""Contact sheets of labeled crops (for the owner to spot mislabels): a grid of crops, each with its crop id."""
from PIL import Image, ImageDraw, ImageFont


def paginate(items: list, per_sheet: int) -> list[list]:
    return [items[i:i + per_sheet] for i in range(0, len(items), per_sheet)]


def contact_sheet(images: list[Image.Image], captions: list[str], title: str, cols: int, cell: tuple[int, int], font_px: int = 20) -> Image.Image:
    """Each image is fitted into a `cell` (w, h) keeping its aspect ratio; the caption sits under it; the title is on top."""
    w, h = cell
    cap_h, title_h = font_px + 10, font_px + 18
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * w, title_h + rows * (h + cap_h)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=font_px)
    d.text((8, 6), title, fill=(0, 0, 0), font=font)
    for i, (im, cap) in enumerate(zip(images, captions)):
        r, c = divmod(i, cols)
        fit = im.copy().convert("RGB")
        fit.thumbnail((w - 8, h - 8))
        x, y = c * w + (w - fit.width) // 2, title_h + r * (h + cap_h) + (h - fit.height) // 2
        sheet.paste(fit, (x, y))
        d.text((c * w + 6, title_h + r * (h + cap_h) + h), cap, fill=(40, 40, 40), font=font)
    return sheet
