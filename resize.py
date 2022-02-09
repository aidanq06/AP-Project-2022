from PIL import Image 

img = Image.open("image.png")
resized = img.resize((28,28))
resized.save("resized_image.jpg")