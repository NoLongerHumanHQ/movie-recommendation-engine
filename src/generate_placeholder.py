#!/usr/bin/env python3
"""
Script to generate a simple placeholder image for movies without posters.
This creates a gray image with text that says "No Poster Available".
"""

from PIL import Image, ImageDraw, ImageFont
import os

def generate_placeholder():
    """Generate a placeholder image for movies without posters"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Path for placeholder image
    placeholder_path = os.path.join(data_dir, "placeholder.jpg")
    
    # Check if placeholder already exists
    if os.path.exists(placeholder_path):
        print(f"Placeholder image already exists at {placeholder_path}")
        return
    
    # Create a gray image
    img = Image.new('RGB', (500, 750), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Add text
    text = "No Poster Available"
    
    # Calculate text position to center it
    textwidth, textheight = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (150, 30)
    x = (500 - textwidth) // 2
    y = (750 - textheight) // 2
    
    # Draw text on image
    draw.text((x, y), text, fill=(200, 200, 200), font=font)
    
    # Save the image
    img.save(placeholder_path, "JPEG")
    print(f"Placeholder image created at {placeholder_path}")

if __name__ == "__main__":
    generate_placeholder() 