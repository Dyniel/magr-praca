from PIL import Image

def create_dummy_image(path, size=(64, 64), color="red"):
    img = Image.new('RGB', size, color=color)
    img.save(path, 'PNG')

if __name__ == "__main__":
    create_dummy_image("dummy_dataset/images/dummy_1.png", color="red")
    create_dummy_image("dummy_dataset/images/dummy_2.png", color="green")
    create_dummy_image("dummy_dataset/images/dummy_3.png", color="blue")
    create_dummy_image("dummy_dataset/images/dummy_4.png", color="black")
    create_dummy_image("dummy_dataset/images/dummy_5.png", color="white")
    create_dummy_image("dummy_dataset/images/dummy_6.png", color="yellow")
    create_dummy_image("dummy_dataset/images/dummy_7.png", color="purple")
    create_dummy_image("dummy_dataset/images/dummy_8.png", color="orange")
    print("Dummy images created successfully.")
