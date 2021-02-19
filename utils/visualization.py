import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotByEpoch(PATH, normalize=True, saveImg=False):
    '''
    visualize the Reduction Rate data with epoch as x axis,
    demonstrate the layer change
    '''

    f = torch.load(PATH, map_location='cpu')

    rate = f['rate']

    x = []
    for i in range(len(f['loss_test'])):
        x.append(np.array(rate[i], dtype=float))
    x = torch.Tensor(x)

    # x = x.permute(1,0,2)
    # n_sample, epochsize, n_rate
    # for t in range(5):
    if normalize:
        x = x.softmax(dim=0).numpy()
    else:
        x = x.numpy()

    columns = ['L'+str(i+1) for i in range(x.shape[1])]

    x = pd.DataFrame(data=x, columns=columns)

    ax = sns.lineplot(data=x)

    if saveImg:
        plt.savefig('img.png', dpi=600)

    plt.show()


    return True

def plotByLayer(PATH, normalize=True, saveImg=False):

    f = torch.load(PATH, map_location='cpu')

    rate = f['rate']
    x = []
    for i in range(len(f['loss_test'])):
        x.append(np.array(rate[i], dtype=float))
    x = torch.Tensor(x)

    if normalize:
        for i in range(1,len(f['loss_test'])):
            x[i] = (x[i] - x[0]) / x[0]
        x[0] = 0
        x = x.numpy()
    else:
        x = x.numpy()

    c_num = x.shape[0]

    columns = ['L' + str(i + 1) for i in range(x.shape[1])]
    x = pd.DataFrame(data=x, columns=columns)

    sns.set_palette(sns.color_palette("Blues", c_num))
    for i in range(c_num):
        ax = sns.lineplot(data=x.iloc[i])

    if saveImg:
        plt.savefig('img.png', dpi=600)
    plt.show()

    return True

def plotList(PATH, yaxis, normalize=True, saveImg=False):
    f = torch.load(PATH, map_location='cpu')
    data = torch.tensor([f[i] for i in yaxis])

    if normalize:
        data = data.softmax(dim=1).numpy().transpose(1,0)
    else:
        data = data.numpy().transpose(1, 0)

    data = pd.DataFrame(data=data, columns=yaxis)

    sns.lineplot(data=data)

    plt.show()
    if saveImg:
        plt.savefig('img.png', dpi=600)
    plt.show()

    return True

def plotMcr2TradeOff(PATH, perLayer=False, saveImg=False):
    f = torch.load(PATH, map_location='cpu')

    rate = np.array([f['rate'][i] for i in range(len(f['acc_test']))], dtype=float) # (n_epoch, n_layer, 2)
    n_epoch = rate.shape[0]
    n_layer = rate.shape[1]

    sns.set_palette(sns.color_palette("Blues", n_epoch))
    if perLayer:
        for j in range(n_layer):
            for i in range(n_epoch):
                sns.scatterplot(x=[rate[i,j,0]], y=[rate[i,j,1]])
            if saveImg:
                plt.savefig('img'+str(j)+'.png', dpi=600)

            plt.show()
    else:
        for i in range(n_epoch):
            sns.scatterplot(x=rate[i,:,0], y=rate[i,:,1])

        if saveImg:
            plt.savefig('img.png', dpi=600)

        plt.show()

    return True


