from UFE import model as m
from UFE import layer as l
from UFE.utilities import img as im
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib

# Load CIFAR-10 Data set from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# The following data format explanation copied from readme.html in CIFAR-10:

# Each of the batch files contains a dictionary with the following elements:
# data -- a 10000x3072 numpy array of uint8s. 
# Each row of the array stores a 32x32 colour image. 
# The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
# The image is stored in row-major order, 
# so that the first 32 entries of the array are the red channel values of the first row of the image.

# labels -- a list of 10000 numbers in the range 0-9.
# The number at index i indicates the label of the ith image in the array data.

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

directory = "./data/cifar-10-batches-py/"
raw_data = unpickle(directory + "data_batch_1")
print(f"Keys: {raw_data.keys()}")
labels = np.array(raw_data[b"labels"])
print(f"label shape: {labels.shape}")
imgs = np.array(raw_data[b"data"]).reshape((10000, 3, 32, 32))

# Util methods

def show_img():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i].T)
    plt.show()

def evaluate_img(model, count=25):
    label_text = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    predicted_label = []
    normalized_img = (imgs / 128) - 1
    for i in range(count):
        output = model.forward(normalized_img[100+i])
        choice = np.argmax(output)
        predicted_label.append(label_text[choice])
    grey_imgs = np.mean(imgs[100:100+count], 1)
    im.show_images(grey_imgs, titles=predicted_label)

# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# input pre-processing
#imgs = (imgs / 128) - 1
print(f"Img: mean={np.mean(imgs)}, max={np.max(imgs)}, min={np.min(imgs)}")

# Set up model
SAVE_PATH = "./UFE/tests/model"

layer = l.CNN(input_shape=(2, 32 - 3 + 1, 32 - 3 + 1), kernel_size=3, num_kernel=20, padding=False)
model = m.model(layers=[
        l.CNN(input_shape=(3, 32, 32), kernel_size=3, num_kernel=2, padding=False),
        layer,
        l.Dense((layer.output_length, 10), activation=l.Activation.Linear)
    ])
#model.save(SAVE_PATH)

model.load(SAVE_PATH)


evaluate_img(model)

input("check")

epoch = 500000

target = []
for i in range(10):
    ith_target = [0] * 10
    ith_target[i] = 1
    target.append(ith_target)

mean_time = 3 # per 100 iterations
last_time = time.time()
def update_mean_time():
    global mean_time
    global last_time
    delta_time = time.time() - last_time
    mean_time = mean_time + 0.1 * (delta_time - mean_time)
    last_time = time.time()

size = len(imgs)-1000
size = 300
choices = [0] * size


try:
    for i in range(epoch):
        for x in range(size):
            output = model.forward(imgs[x])
            model.backprop_target(target[labels[x]], 0.001)

            output_choice = np.argmax(output)
            if (x % 100 == 0):
                print(f"accuracy: {np.round((np.sum(choices) / size) * 100, 2)}% choice: {output_choice} label: {labels[x]}, output: {np.round(output, 2)}")

            if (output_choice == labels[x]):
                choices[x] = 1
            else:
                choices[x] = 0

            if (np.amax(output) > 100):
                raise Exception(f"Too big output max: {np.amax(output)}")
except KeyboardInterrupt:
    model.save(SAVE_PATH)
    pass

