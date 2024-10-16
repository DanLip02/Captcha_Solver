import random
from captcha.image import ImageCaptcha
import string
import random
from PIL import ImageDraw, Image
# string.ascii_uppercase
# string.digits
# !~
ALL_SYMBOLS = string.ascii_uppercase  + string.digits


def gen_captcha_text(length=5):
    symbols_list = []
    for _ in range(length):
        symbols_list.append(random.choice(ALL_SYMBOLS))

    return "".join(symbols_list)

def add_background_noise(image, noise_level=30):
    width, height = image.shape
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            # Добавление шума к каждому цвету пикселя
            noise = random.randint(-noise_level, noise_level)
            r = max(0, min(255, r + noise))
            g = max(0, min(255, g + noise))
            b = max(0, min(255, b + noise))
            image.putpixel((x, y), (r, g, b))


# def add_noise_points(image, num_points=100):
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#
#     for _ in range(num_points):
#         # Генерация случайных координат для точек
#         x, y = random.randint(0, width), random.randint(0, height)
#         # Генерация случайного цвета точки
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         # Рисование точки
#         draw.point((x, y), fill=color)


for _ in range(1500):
    # Image captcha text
    captcha_text = gen_captcha_text(length=5)
    # generate the image of the given text
    # Create an image instance of the given size
    mylist = [i for i in range(10, 60)]
    number_1 = random.choice(mylist)
    number_2 = random.choice(mylist)
    number_3 = random.choice(mylist)
    image = ImageCaptcha(width=280, height=80, font_sizes=(number_1, number_2, number_3))
    print(number_1, number_2)
    data = image.generate(captcha_text)
    image.character_rotate = (number_1, number_2)
    # Добавление точек шума
    # add_noise_points(image, num_points=500)
    # write the image on the given file and save it
    image.write(captcha_text, f'generator_2_upd/{captcha_text}.png')