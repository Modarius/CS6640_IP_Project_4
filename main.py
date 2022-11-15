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
    mask = lpfMask(img1, radius=100)

    top_fract = np.multiply(fft_img1, fft_img2)
    bot_fract = np.abs(top_fract)
    top_fract = np.fft.fftshift(top_fract)
    top_fract = np.fft.ifftshift(np.multiply(mask, top_fract))

    pxy = np.fft.ifft2(np.divide(top_fract, bot_fract))
    return pxy

def peakFind(in_img):
    width = in_img.shape[0]
    height = in_img.shape[1]
    idx = np.argmax(in_img)
    x = int(idx % in_img.shape[0])
    y = int(idx / in_img.shape[1])
    peaks = [(x,y), (width - x,y), (x, height- y), (width - x, height - y)]
    return peaks

# overlap img2 onto img1, return the result and mse of the overlap
def overlap(img2, img1, offset):
    width = img1.shape[0]
    height = img1.shape[1]
    if (offset[0] < 0):
        if (offset[1] < 0): # overlap is top left corner of img1 with bottom right corner of img2
            a = img1[0:(height - offset[1]), 0:(width - offset[0])]
            b = img2[offset[1]-1:-1, offset[0]-1:-1]
        else:               # overlap is bottom left corner of img1 with top right corner of img2
            a = img1[offset[1]-1:-1, 0:(width - offset[0])]
            b = img2[0:(height - offset[1]), offset[0]-1:-1]
    else:
        if (offset[1] < 0): # overlap is top right corner of img1 with bottom left corner of img2
            a = img1[0:(height - offset[1]), offset[0]-1:-1]
            b = img2[offset[1]-1:-1, 0:(width - offset[0])]
        else:               # overlap is bottom right corner of img1 with top left corner of img2
            a = img1[offset[1]-1:-1, offset[0]-1:-1]
            b = img2[0:(height - offset[1]), 0:(width - offset[0])]
    f = plt.figure()
    f.add_subplot(121)
    io.imshow(a)
    f.add_subplot(122)
    io.imshow(b)
    return

def main():
    image_names = open("lnis-mosaic/read.txt","r").readlines()
    f, image_names = zip(*[('lnis-mosaic/' + fs.strip(), fs.strip().split('.')[0]) for fs in image_names]) #https://stackoverflow.com/a/2050649, https://stackoverflow.com/a/7558990
    
    raw_img = np.empty(len(image_names), dtype=object)
    fft_img = np.empty(len(image_names), dtype=object)
    psd_fft_img = np.empty(len(image_names), dtype=object)
    lpf_fft_img = np.empty(len(image_names), dtype=object)
    lpf_img = np.empty(len(image_names), dtype=object)
    for i in range(len(image_names)):
        raw_img[i] = io.imread(f[i], as_gray=True)
        fft_img[i] = np.fft.fftshift(np.fft.fft2(raw_img[i])) # shifts corners of 2D fft to center
        psd_fft_img[i] = np.log10(np.abs(fft_img[i])**2) # https://dsp.stackexchange.com/a/10064
        lpf_fft_img[i] = np.multiply(lpfMask(fft_img[i]), fft_img[i]) 
        lpf_img[i] = np.fft.ifft2(np.fft.ifftshift(lpf_fft_img[i])).real
    #plot_array(psd_fft_img, image_names)
    #plot_array(lpf_img, image_names)

    pxy = phase_correlation(raw_img[0], raw_img[1]).real
    peaks = peakFind(pxy)
    #io.imshow(pxy)
    for peak in peaks:
        img1 = raw_img[0]
        img2 = raw_img[1]
        overlap(img1, img2, peak)
    return

if __name__ == "__main__":
    main()