from skimage.color import rgb2ycbcr, ycbcr2rgb
import numpy as np
import scipy.misc
import cv2
from scipy.fftpack import dct,idct

w = 8  # Thay đổi nếu cần, nhưng giữ nằm giữa 2 và 8 do bảng lượng tử hóa mặc định là 8x8
w = max(2, min(8, w))  # Đảm bảo w nằm trong khoảng từ 2 đến 8

# Đặt chiều cao khối bằng chiều rộng khối
h = w

def myYcbcr2rgb(ycbcr):
    # Chuyển đổi YCbCr sang RGB và giới hạn giá trị nằm trong khoảng [0, 1]
    return (ycbcr2rgb(ycbcr).clip(0, 1) * 255).astype(np.uint8)

def toBlocks(img):
    xLen = img.shape[1]//w
    yLen = img.shape[0]//h
    blocks = np.zeros((yLen,xLen,h,w,3),dtype=np.int16)
    for y in range(yLen):
        for x in range(xLen):
            blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
    return np.array(blocks)

# def plot_blocks(blocks, gray=False):
#     y_len, x_len, _, _, _ = blocks.shape
    
#     for y in range(y_len):
#         for x in range(x_len):
#             plt.subplot(y_len, x_len, 1 + x_len * y + x)
#             plt.imshow(blocks[y][x], cmap='gray' if gray else 'Accent')
#             plt.axis('off')

def dctOrDedctAllBlocks(img, blocks,type="dct"):
    xLen = img.shape[1]//w
    yLen = img.shape[0]//h
    f=dct if type=="dct" else idct
    dedctBlocks = np.zeros((yLen,xLen,h,w,3))
    for y in range(yLen):
        for x in range(xLen):
            d = np.zeros((h,w,3))
            for i in range(3):
                block=blocks[y][x][:,:,i]
                d[:,:,i]=f(f(block.T, norm = 'ortho').T, norm = 'ortho')
                if (type!="dct"):
                    d=d.round().astype(np.int16)
            dedctBlocks[y][x]=d
    return dedctBlocks
def blocks2img(img, blocks):
    xLen = img.shape[1]//w
    yLen = img.shape[0]//h
    W=xLen*w
    H=yLen*h
    img1 = np.zeros((H,W,3))
    for y in range(yLen):
        for x in range(xLen):
            img1[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
    return img1
