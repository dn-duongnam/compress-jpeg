from util import MultiPage
from app import *

if __name__ == '__main__':
    app = MultiPage()
    app.add_app('Color Image Jpeg',color_image_jpeg)
    app.add_app('Gray Image Jpeg',gray_image_jpeg)
    app.add_app("Denoising Image", denoising_image)
    app.run()
