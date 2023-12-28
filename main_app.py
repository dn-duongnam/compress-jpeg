import streamlit as st
import os
import rawpy
import numpy as np
from skimage import io
from io import BytesIO
import plotly.express as px
from skimage.color import rgb2ycbcr,ycbcr2rgb
from scipy.fftpack import dct,idct
from copy import copy
import plotly.figure_factory as ff
from scipy.fftpack import dct,idct
import plotly.graph_objects as go
import huffman
from collections import Counter
import time
import cv2
from util import MultiPage
from app import *

if __name__ == '__main__':
    app = MultiPage()
    app.add_app('Color Image Jpeg',color_image_jpeg)
    app.add_app('Gray Image Jpeg',gray_image_jpeg)
    app.run()
