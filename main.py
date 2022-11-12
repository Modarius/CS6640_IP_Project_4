import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as filt
import skimage.draw as drw
from skimage import io
from skimage.util import img_as_uint

def lpfMask(in_img, radius=100):
    img_width = in_img.shape[-2]
    img_height = in_img.shape[-1]
    rr, cc = drw.disk((int(img_width/2), int(img_height/2)), radius=radius, shape=(img_width, img_height))
    mask = np.zeros((img_width, img_height))
    mask[rr, cc] = 1
    return mask

def plot_array(imgs, names):
    figure = io.imshow_collection(imgs) # show the collection of processed ffts
    plt.setp(figure.axes, xticks=[], yticks=[]) # https://stackoverflow.com/a/54519425 how to get rid of axes ticks
    for i in range(len(names)): figure.axes[i].set_title(names[i]) # add title strings to each image
    return figure

def phase_correlation(img1, img2):
    fft_img1 = np.fft.fft2(img1)
    fft_img2 = np.fft.fft2(img2).conjugate()
    mask = lpfMask(img1, radius=20)

    top_fract = np.multiply(fft_img1, fft_img2)
    bot_fract = np.abs(top_fract)
    top_fract = np.fft.fftshift(top_fract)
    top_fract = np.fft.ifftshift(np.multiply(mask, top_fract))

    pxy = np.fft.ifft2(np.divide(top_fract, bot_fract))
    return pxy

def peakFind(in_img):
    idx = np.argmax(in_img)
    x = idx % in_img.shape[0]
    y = idx / in_img.shape[1]
    return x, y

def main():
    image_names = open("cell_images/read.txt","r").readlines()
    f, image_names = zip(*[('cell_images/' + fs.strip(), fs.strip().split('.')[0]) for fs in image_names]) #https://stackoverflow.com/a/2050649, https://stackoverflow.com/a/7558990
    raw_img = io.imread_collection(f)
    fft_img = np.fft.fftshift(np.fft.fft2(raw_img)) # shifts corners of 2D fft to center
    psd_fft_img = np.log(np.abs(fft_img)**2) # https://dsp.stackexchange.com/a/10064
    #plot_array(psd_fft_img, image_names)

    lpf_fft_img = np.multiply(lpfMask(fft_img), fft_img[:])
    lpf_img = np.fft.ifft2(np.fft.ifftshift(lpf_fft_img)).real
    #plot_array(lpf_img.astype(np.int8), image_names)

    pxy = phase_correlation(raw_img[2], raw_img[2]).real
    x, y = peakFind(pxy)
    io.imshow(pxy)
    return

if __name__ == "__main__":
    main()