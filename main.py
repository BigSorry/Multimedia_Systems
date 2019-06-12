import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import glob
import copy
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

path = "images/*.jpg"
imageList = []
for filename in glob.glob(path):
    imageList.append(Image.open(filename))

imgSize = 128
transform = transforms.Compose([
    transforms.Resize((imgSize,imgSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
def imToTrain(im):
    im = transform(im)
    im = im.unsqueeze(0)
    im = Variable(im)
    return im

def makeLayers(cnn, last = 5):
    model = nn.Sequential()
    i = 0
    name = ""
    sequences = list(cnn.children())[0]
    for layer in sequences:
        if isinstance(layer, nn.Conv2d):
            i += 1
            if i > last:
                break
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, nn.Sequential):
             name = 'seq_{}'.format(i)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            name = 'avgpool_{}'.format(i)
        elif isinstance(layer, nn.Linear):
            name = 'linear_{}'.format(i)
        else:
            continue
        model.add_module(name, layer)
    return model

def plot(matches):
    plt.rcParams['axes.linewidth'] = 3
    plt.figure()
    plt.imshow(imageList[0])
    plt.title("Search image")
    plt.xticks([])
    plt.yticks([])

    plt.figure()
    plt.title("Matches")
    rows = int(np.ceil(len(matches) / 2))
    for key, value in matches.items():
        ax = plt.subplot(2, rows, key)
        color = 'red'
        if value == True:
            color = "green"
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        plt.imshow(imageList[key])
        plt.xticks([])
        plt.yticks([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = models.vgg19_bn(pretrained=True).eval()
cnn = copy.deepcopy(pretrained)
model = makeLayers(cnn)

searchImg = imToTrain(imageList[0])
searchEncoding = model(searchImg).flatten().detach().numpy()
searchEncoding2 = model(searchImg).detach()
matches = {i:False for i in range(1, len(imageList))}
threshold = 0.01
for i in range(1, len(imageList)):
    # compareEncoding = model(imToTrain(imageList[i])).flatten().detach().numpy()
    # mse = sum(abs(compareEncoding - searchEncoding)**2)
    compareEncoding2 = model(imToTrain(imageList[i])).detach()
    normalizedMse = F.mse_loss(searchEncoding2, compareEncoding2)

    value = normalizedMse.numpy()
    print("Mse is {}".format(value))
    if value < threshold:
        matches[i] = True
    if i % 10 == 0:
        print("At image {}".format(i))

print("Done matching, now plotting")
plot(matches)
plt.show()