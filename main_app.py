from util import MultiPage
from app import *

if __name__ == '__main__':
    app = MultiPage()
    app.add_app('Color Image Jpeg',color_image_jpeg)
    app.add_app('Gray Image Jpeg',gray_image_jpeg)
    app.add_app("Denoising Image (Weight Mean)", denoising_image)
    app.add_app("Denoising Image (Gaussian Importance Map)", denoising_image_gaussian)
    app.add_app("Denoising Image Gray", denoising_image_gray)
    app.add_app("Denoising Periodic", denoising_periodic)
    app.run()
