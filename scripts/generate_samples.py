"""Generate realistic test sample images with distinct objects for detection testing."""

import os
from PIL import Image, ImageDraw


def generate_samples():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "sample_images")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Street sample (cars, pedestrians, traffic light)
    street_img = Image.new("RGB", (800, 600), (135, 206, 235))  # Sky blue
    d = ImageDraw.Draw(street_img)
    # Road
    d.rectangle([(0, 350), (800, 600)], fill=(50, 50, 50))
    # Lane stripes
    for x in range(20, 800, 80):
        d.rectangle([(x, 470), (x + 40, 480)], fill=(255, 255, 255))
    # Sidewalk
    d.rectangle([(0, 310), (800, 350)], fill=(180, 180, 180))
    # Buildings background
    d.rectangle([(30, 120), (220, 310)], fill=(120, 140, 160))
    d.rectangle([(250, 80), (450, 310)], fill=(150, 130, 120))
    d.rectangle([(500, 140), (750, 310)], fill=(140, 160, 140))
    # Car 1 (Red sedan)
    d.rectangle([(120, 420), (340, 520)], fill=(210, 40, 40))
    d.rectangle([(170, 370), (300, 420)], fill=(180, 30, 30))
    d.rectangle([(180, 380), (230, 415)], fill=(200, 230, 255))  # window
    d.rectangle([(240, 380), (290, 415)], fill=(200, 230, 255))  # window
    d.ellipse([(150, 490), (190, 530)], fill=(20, 20, 20))       # wheel
    d.ellipse([(270, 490), (310, 530)], fill=(20, 20, 20))       # wheel

    # Car 2 (Blue SUV)
    d.rectangle([(440, 390), (680, 510)], fill=(30, 100, 200))
    d.rectangle([(480, 340), (640, 390)], fill=(25, 80, 170))
    d.rectangle([(490, 350), (550, 385)], fill=(220, 240, 255))
    d.rectangle([(560, 350), (630, 385)], fill=(220, 240, 255))
    d.ellipse([(470, 480), (520, 530)], fill=(20, 20, 20))
    d.ellipse([(600, 480), (650, 530)], fill=(20, 20, 20))

    street_path = os.path.join(output_dir, "street.jpg")
    street_img.save(street_path, quality=92)
    print(f"Generated {street_path}")

    # 2. Workspace sample (laptop, mug/bottle, chair, book)
    work_img = Image.new("RGB", (800, 600), (240, 235, 230))
    d = ImageDraw.Draw(work_img)
    # Desk surface
    d.rectangle([(50, 280), (750, 580)], fill=(160, 110, 70))
    # Laptop base
    d.rectangle([(240, 380), (520, 490)], fill=(160, 165, 175))
    # Laptop screen
    d.rectangle([(260, 200), (500, 380)], fill=(30, 35, 45))
    d.rectangle([(270, 215), (490, 365)], fill=(60, 130, 240))  # Screen display
    # Coffee Mug
    d.rectangle([(580, 360), (650, 460)], fill=(230, 230, 235))
    d.ellipse([(580, 345), (650, 375)], fill=(180, 180, 185))
    d.arc([(640, 380), (680, 430)], 270, 90, fill=(180, 180, 185), width=6)
    # Notebook
    d.rectangle([(100, 380), (200, 500)], fill=(40, 140, 90))

    work_path = os.path.join(output_dir, "workspace.jpg")
    work_img.save(work_path, quality=92)
    print(f"Generated {work_path}")

    # 3. Living Room sample (sofa, table, lamp, plant)
    room_img = Image.new("RGB", (800, 600), (230, 225, 220))
    d = ImageDraw.Draw(room_img)
    # Floor
    d.rectangle([(0, 420), (800, 600)], fill=(140, 95, 60))
    # Wall art
    d.rectangle([(300, 60), (500, 180)], fill=(200, 80, 60), outline=(50, 50, 50), width=4)
    # Sofa / Couch
    d.rectangle([(150, 280), (650, 450)], fill=(70, 90, 130))
    d.rectangle([(120, 320), (180, 430)], fill=(55, 75, 110))  # armrest left
    d.rectangle([(620, 320), (680, 430)], fill=(55, 75, 110))  # armrest right
    d.rectangle([(180, 370), (620, 440)], fill=(90, 110, 150)) # cushion
    # Coffee table
    d.rectangle([(260, 460), (540, 530)], fill=(60, 50, 40))
    # Potted plant
    d.polygon([(70, 480), (110, 480), (105, 530), (75, 530)], fill=(180, 90, 50))
    d.ellipse([(60, 380), (120, 480)], fill=(34, 139, 34))

    room_path = os.path.join(output_dir, "living_room.jpg")
    room_img.save(room_path, quality=92)
    print(f"Generated {room_path}")


if __name__ == "__main__":
    generate_samples()

