import os
from PIL import Image

def create_dummy_image(path, color="red"):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  img = Image.new('RGB', (60, 30), color=color)
  img.save(path, 'PNG')

if __name__ == "__main__":
    create_dummy_image("dummy_dataset/images/dummy_1.png", color="red")
    create_dummy_image("dummy_dataset/images/dummy_2.png", color="blue")
