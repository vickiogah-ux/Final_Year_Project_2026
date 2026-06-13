import os
from PIL import Image, ImageDraw

def create_icon(size):
    os.makedirs("extension/icons", exist_ok=True)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    padding = size * 0.1
    left = padding
    right = size - padding
    top = padding
    bottom = size - padding
    middle_x = size / 2
    
    y1 = top
    y2 = top + (bottom - top) * 0.15
    y3 = top + (bottom - top) * 0.65
    y4 = bottom
    
    points = [
        (middle_x, y1),
        (right, y2),
        (right, y3),
        (middle_x, y4),
        (left, y3),
        (left, y2)
    ]
    
    # Draw royal purple shield with cyan border
    draw.polygon(points, fill=(168, 85, 247, 255), outline=(6, 182, 212, 255), width=max(1, int(size * 0.06)))
    
    # Draw checkmark inside for 48 and 128
    if size >= 48:
        cx1 = middle_x - size * 0.15
        cy1 = middle_x + size * 0.05
        cx2 = middle_x - size * 0.05
        cy2 = middle_x + size * 0.15
        cx3 = middle_x + size * 0.18
        cy3 = middle_x - size * 0.12
        draw.line([(cx1, cy1), (cx2, cy2), (cx3, cy3)], fill=(6, 182, 212, 255), width=max(2, int(size * 0.06)))
        
    img.save(f"extension/icons/icon{size}.png")
    print(f"SUCCESS: Created icon{size}.png")

if __name__ == "__main__":
    create_icon(16)
    create_icon(48)
    create_icon(128)
