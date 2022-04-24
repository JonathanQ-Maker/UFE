import matplotlib.pyplot as plt
import numpy as np


def get_img_from_kspace(fft_img):
    """
    given 2d kspace return [0, 255] image of same size
    @params:
        fft_img - Required : matrix of shape (height, width)
    """

    img = np.fft.ifft2(fft_img)
    img = np.sqrt(img.real * img.real + img.imag * img.imag)
    img = img / np.amax(img)
    return img * 255

def show_images(imgs, mode="grid", titles=None):
    """
    display gray scale images
    @params:
        imgs - Required : matrix of shape (# of imgs, height, width)
        mode   - Optional: 
                    "grid" for grid layout 
                    "click" for click to cycle layout
    """
    if (mode == "grid"):
        n: int = len(imgs)
        sqrt_n = np.ceil(np.sqrt(n)).astype(int)
        f = plt.figure()
        
        for i in range(n):
            # Debug, plot figure
            plot = f.add_subplot(sqrt_n, sqrt_n, i + 1)
            if (titles == None):
                plot.title.set_text(f'index: {i}')
            else:
                plot.title.set_text(titles[i])
            plt.imshow(np.uint8(imgs[i]), cmap=plt.get_cmap('gray'))
        plt.subplots_adjust(hspace =0.5)
        plt.show(block=True)

    elif (mode == "click"):
        cycler = ImgCycler(imgs)
        cycler.show_img()

class ImgCycler:
    """
    Display images with added event to cycle images on click
    """

    def __init__(self, imgs):
        """
        @params:
            imgs - Required: matrix of shape (# of imgs, height, width)
        """

        self.imgs = imgs
        self.index = 0
        self.max_index = len(self.imgs)

    def onclick(self, event):
        """
        Do not use individually, event callback function
        """

        self.index += 1
        if (self.index >= self.max_index):
            self.index = 0

        event.canvas.figure.clear()
        event.canvas.figure.suptitle(f'Index: {self.index}', fontsize=12)
        event.canvas.figure.gca().imshow(self.imgs[self.index], cmap=plt.get_cmap('gray'))
        event.canvas.draw()

    def show_img(self):
        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.imshow(self.imgs[0], cmap=plt.get_cmap('gray'))
        plt.show()

def show_kspace(kspace, mode="grid"):
    """
    display image generated from kspace
    @params:
        kspace - Required: matrix shape (# of imgs, height, width)
        mode   - Optional: 
                    "grid" for grid layout 
                    "click" for click to cycle layout
    """

    imgs = []
    for slice in kspace:
        img = np.fft.fftshift(get_img_from_kspace(slice))
        imgs.append(img)
    imgs = np.array(imgs)
    show_images(imgs, mode)

def complex_to_3d(complex_img):
    return [complex_img.real, complex_img.imag]

def real_to_complex(real):
    return 1j*real[1, :,:] + real[0, :,:]