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

path = "images/database/*.jpg"
imageList = []
for filename in glob.glob(path):
    imageList.append(Image.open(filename))

path = "images/search/*.jpg"
searchImage = []
for filename in glob.glob(path):
    searchImage = Image.open(filename)
    
imgSize = 56
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

def getMatches(cnn, search, targets, layers):
    model = nn.Sequential()
    i = 0
    name = ""
    sequences = list(cnn.children())[0]
    mseValues = [0 for i in range(0, len(targets))]
    threshold = 0.01
    for layer in sequences:
        if isinstance(layer, nn.Conv2d):
            i += 1
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
        if name in layers:
            searchEncoding = model(search).detach()
            for i in range(0, len(imageList)):
                compareEncoding = model(targets[i]).detach()
                normalizedMse = F.mse_loss(searchEncoding, compareEncoding)

                mseValues[i] += normalizedMse.numpy()
                print("Mse is {}".format(mseValues[i]))

    result = {i: False for i in range(0, len(targets))}
    for i in range(0, len(mseValues)):
        if mseValues[i]/len(layers) < threshold:
            result[i] = True

    return result

def plot(matches):
    plt.rcParams['axes.linewidth'] = 3
    plt.figure()
    rows = int(np.ceil(len(matches) / 2)) + 1
    ax = plt.subplot(2,rows, 1)
    ax.spines['bottom'].set_color("black")
    ax.spines['top'].set_color("black")
    ax.spines['right'].set_color("black")
    ax.spines['left'].set_color("black")
    plt.imshow(searchImage)
    plt.xticks([])
    plt.yticks([])

    for key, value in matches.items():
        ax = plt.subplot(2, rows, key+1)
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
    plt.savefig('result/matches.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = models.vgg19_bn(pretrained=True).eval()
cnn = copy.deepcopy(pretrained)

layers = ['conv_4', 'conv_5']
targets = [imToTrain(image) for image in imageList]
matches = getMatches(cnn, imToTrain(searchImage), targets, layers)
print("Done matching, now plotting")
plot(matches)
