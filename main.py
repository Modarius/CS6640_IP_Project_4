from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as drw
from skimage import io
from skimage import metrics as met

def plotFour(img1, img2, img3, img4):
    f = plt.figure()
    f.add_subplot(221)
    io.imshow(img1)
    f.add_subplot(222)
    io.imshow(img2)
    f.add_subplot(223)
    io.imshow(img3)
    f.add_subplot(224)
    io.imshow(img4)
    return

def lpfMask(in_img, radius=100):
    img_width = in_img.shape[-1]
    img_height = in_img.shape[-2]
    rr, cc = drw.disk((int(img_height/2), int(img_width/2)), radius=radius, shape=(img_height, img_width)) # https://scikit-image.org/docs/stable/api/skimage.draw.html?highlight=mask#skimage.draw.disk
    mask = np.zeros((img_height, img_width))
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

def findOffsets(in_img):
    w = in_img.shape[-1]
    h = in_img.shape[-2]
    idx = np.argmax(in_img)
    y,x = np.unravel_index(idx, in_img.shape) # https://stackoverflow.com/a/47726092
    offsets = [(y, x), (y, x - w), (y - h, x), (y - h, x - w)] # note that these indices are flipped to match np array indexing which is row (y) column (x)
    return offsets

def subImage(img1, img2, offset):
    w = img1.shape[-1]
    h = img1.shape[-2]
    x = offset[1]
    y = offset[0]

    # I consider img1 to be the image on the bottom of the overlap, the offset to be from img1 (0,0) to img2 (0,0)
    if (x > 0):
        if (y > 0): # overlap is b-r of img1, t-l of img2
            x, y = abs(x), abs(y)
            a = img1[y:,x:]
            b = img2[0:h-y, 0:w-x]
        else:       # overlap is t-r of img1, b-l of img2
            x, y = abs(x), abs(y)
            a = img1[0:h-y, x:]
            b = img2[y:,0:w-x]
    else:
        if (y > 0): # overlap is b-l of img1, t-r of img2
            x, y = abs(x), abs(y)
            a = img1[y:, 0:w-x]
            b = img2[0:h-y, x:]
        else:       # overlap is t-l of img1, b-r of img2
            x, y = abs(x), abs(y)
            a = img1[0:h-y, 0:w-x]
            b = img2[y:, x:]
    return a, b

# determine which offset is most likely to be correct
def bestOffset(img1, img2, offsets, show_plots=False):
    ss = dict()
    for offset in offsets:
        a, b = subImage(img1, img2, offset)
        if(a.shape[0] > 10 and a.shape[1] > 10):
            b = np.where(a == 0, 0, b)
            a = np.where(b == 0, 0, a)
            ss[met.structural_similarity(a, b)] = offset # uses different algorithm to determine if images are simular, better for small alignment errors
            if(show_plots): 
                plotFour(img1, img2, a, b)
    max_similarity = np.max(list(ss.keys()))
    if (max_similarity > .8):
        best_offset = ss[max_similarity]
    else:
        best_offset = None
    return best_offset, max_similarity

def overlap(img1, img2, offset):
    w = img1.shape[-1]
    h = img1.shape[-2]
    x = offset[1]
    y = offset[0]
    canvas = np.zeros((h+abs(y), w+abs(x)))
    # I consider img1 to be the image on the bottom of the overlap, the offset to be from img1 (0,0) to img2 (0,0)
    if(x > 0):
        if(y > 0):  # overlap is b-r of img1, t-l of img2
            x, y = abs(x), abs(y)
            canvas[0:h, 0:w] = img1
            canvas[y:y+h, x:x+w] = img2
        else:       # overlap is t-r of img1, b-l of img2
            x, y = abs(x), abs(y)
            canvas[y:y+h, 0:w] = img1
            canvas[0:h, x:x+w] = img2
    else:
        if(y > 0):  # overlap is b-l of img1, t-r of img2
            x, y = abs(x), abs(y)
            canvas[0:h, x:x+w] = img1
            canvas[y:y+h, 0:w] = img2
        else:       # overlap is t-l of img1, b-r of img2
            x, y = abs(x), abs(y)
            canvas[y:y+h, x:x+w] = img1
            canvas[0:h, 0:w] = img2
    return canvas

# this code assumes that all images have some overlap with at least one other image
def mosaic(imgs):
    base_img = imgs.pop() # select an initial image to build the mosaic from
    w = base_img.shape[-1]
    h = base_img.shape[-2]
    while (len(imgs) > 0):
        img_data = dict()
        for i in range(len(imgs)):
            curr_img = np.zeros(base_img.shape)
            curr_img[0:h,0:w] = imgs[i]
            pxy = phase_correlation(curr_img, base_img).real
            offsets = findOffsets(pxy)
            best_offset, similarity = bestOffset(img1=curr_img, img2=base_img, offsets=offsets, show_plots=True)
            plt.close('all')
            if best_offset is not None:
                img_data[similarity] = (i,  best_offset)
        image_idx, img_offset = img_data[np.max(list(img_data))]
        best_img = np.zeros(base_img.shape)
        best_img[0:h,0:w] = imgs.pop(image_idx)
        base_img = overlap(img1=best_img, img2=base_img, offset=img_offset)
        io.imshow(base_img)
    return base_img


def main():
    base_path = "lnis-mosaic/"
    image_names = open(base_path + "read.txt","r").readlines()
    f, image_names = zip(*[(base_path + fs.strip(), fs.strip().split('.')[0]) for fs in image_names]) #https://stackoverflow.com/a/2050649, https://stackoverflow.com/a/7558990
    
    raw_imgs = np.empty(len(image_names), dtype=object)
    fft_img = np.empty(len(image_names), dtype=object)
    psd_fft_img = np.empty(len(image_names), dtype=object)
    lpf_fft_img = np.empty(len(image_names), dtype=object)
    lpf_img = np.empty(len(image_names), dtype=object)
    raw_imgs = list()
    for i in range(len(image_names)):
        raw_imgs.append(io.imread(f[i], as_gray=True))
        fft_img[i] = np.fft.fftshift(np.fft.fft2(raw_imgs[i])) # shifts corners of 2D fft to center
        psd_fft_img[i] = np.log10(np.abs(fft_img[i])**2) # https://dsp.stackexchange.com/a/10064
        lpf_fft_img[i] = np.multiply(lpfMask(fft_img[i]), fft_img[i]) 
        lpf_img[i] = np.fft.ifft2(np.fft.ifftshift(lpf_fft_img[i])).real
    #plot_array(psd_fft_img, image_names)
    #plot_array(lpf_img, image_names)

    #io.imshow(pxy)
    output = mosaic(raw_imgs)
    io.imsave(base_path + "output.png", output)
    io.imshow(output)
    return

if __name__ == "__main__":
    main()