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

    if saveImg:
        plt.savefig('img.png', dpi=600)

    plt.show()

    return True

def plotMcr2TradeOff(PATH, To=-1, perLayer=False, saveImg=False):
    f = torch.load(PATH, map_location='cpu')

    rate = np.array([f['rate'][i] for i in range(len(f['acc_test']))], dtype=float) # (n_epoch, n_layer, 2)
    n_epoch = rate.shape[0]
    if To > -1:
        n_epoch = To

    n_layer = rate.shape[1]

    # sns.set_palette(sns.color_palette("blue", n_epoch))
    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
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

def saveCaL(PATH, perLayer, name):
    f = torch.load(PATH, map_location='cpu')
    yaxis_1 = ['acc_test', 'acc_train']
    yaxis_2 = ['loss_test', 'loss_train']

    data = torch.tensor([f[i] for i in yaxis_1])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_1)

    sns.lineplot(data=data)

    plt.savefig('Img/CaL/acc/'+name+'_acc.png', dpi=600)

    plt.show()

    data = torch.tensor([f[i] for i in yaxis_2])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_2)

    sns.lineplot(data=data)

    plt.savefig('Img/CaL/loss/' + name + '_loss.png', dpi=600)

    plt.show()

    rate = np.array([f['rate'][i] for i in range(len(f['acc_test']))], dtype=float)  # (n_epoch, n_layer, 2)
    crate = np.array([f['crate'][i] for i in range(len(f['acc_test']))], dtype=float)

    n_epoch = rate.shape[0]

    n_layer = rate.shape[1]

    # sns.set_palette(sns.color_palette("blue", n_epoch))
    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    # plot rate
    if perLayer:
        for j in range(n_layer):
            for i in range(n_epoch):
                sns.scatterplot(x=[rate[i, j, 0]], y=[rate[i, j, 1]])
            plt.savefig('Img/CaL/rate/'+name+ '_rate_'+str(j) + '.png', dpi=600)

            plt.show()
    else:
        for i in range(n_epoch):
            sns.scatterplot(x=rate[i, :, 0], y=rate[i, :, 1])

        plt.savefig('Img/CaL/rate/'+name+'_rate.png', dpi=600)

        plt.show()
    # plot crate
    if perLayer:
        for j in range(n_layer):
            for i in range(n_epoch):
                sns.scatterplot(x=[crate[i, j, 0]], y=[crate[i, j, 1]])
            plt.savefig('Img/CaL/crate/'+name+ '_crate_'+str(j) + '.png', dpi=600)

            plt.show()
    else:
        for i in range(n_epoch):
            sns.scatterplot(x=crate[i, :, 0], y=crate[i, :, 1])

        plt.savefig('Img/CaL/crate/'+name+'_crate.png', dpi=600)

        plt.show()

    return True

def saveCaL_rateDistortion(PATH, perLayer, name):
    f = torch.load(PATH, map_location='cpu')
    yaxis_1 = ['acc_test', 'acc_train']
    yaxis_2 = ['loss_test', 'loss_train']

    data = torch.tensor([f[i] for i in yaxis_1])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_1)

    sns.lineplot(data=data)

    plt.savefig('Img/CaL_rateDistortion/' + name + '/acc/' + 'acc.png', dpi=600)

    plt.show()

    data = torch.tensor([f[i] for i in yaxis_2])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_2)

    sns.lineplot(data=data)

    plt.savefig('Img/CaL_rateDistortion/' + name + '/loss/' + 'loss.png', dpi=600)

    plt.show()

    rate = f['rate']

    n_epoch = rate.shape[0]
    n_layer = rate.shape[1]

    # sns.set_palette(sns.color_palette("blue", n_epoch))
    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    # plot rate
    if perLayer:

        for j in range(n_layer):
            for i in range(n_epoch):
                sns.scatterplot(x=[rate[i, j, 0]], y=[rate[i, j, 1]])
            plt.savefig('Img/CaL_rateDistortion/' + name +'/rate/' + 'mcrd_' + str(j) + '.png', dpi=600)

            plt.show()
    else:


        for i in range(n_epoch):
            sns.scatterplot(x=rate[i, :, 0], y=rate[i, :, 1])

        plt.savefig('Img/CaL_rateDistortion/' + name +'/rate/'+'mcrd.png', dpi=600)

        plt.show()


    return True

def saveIB(PATH, name):
    f = torch.load(PATH, map_location='cpu')

    yaxis_1 = ['acc_test', 'acc_train']
    yaxis_2 = ['loss_test', 'loss_train']
    # -------------------------save acc-------------------------------
    data = torch.tensor([f[i] for i in yaxis_1])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_1)

    sns.lineplot(data=data)

    plt.savefig('Img/forpaper/IB/' + name + '/acc/' + 'acc.png', dpi=600)
    plt.close()
    # -------------------------save loss-------------------------------
    data = torch.tensor([f[i] for i in yaxis_2])
    data = data.numpy().transpose(1, 0)
    data = pd.DataFrame(data=data, columns=yaxis_2)

    sns.lineplot(data=data)

    plt.savefig('Img/forpaper/IB/' + name + '/loss/' + 'loss.png', dpi=600)

    plt.close()

    # -------------------------save global IB-------------------------------

    XT = f['rate_train'][:, :, 0]
    TY = f['rate_train'][:, :, 1]

    n_epoch = XT.shape[0]
    n_layer = XT.shape[1]

    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    for i in range(n_layer):
        for j in range(n_epoch):
            sns.scatterplot(x=[XT[j,i]], y=[TY[j,i]])
            plt.xlabel('I(X;T)')
            plt.ylabel('I(T;Y)')

    plt.savefig('Img/forpaper/IB/' + name + '/IB_train/' + name+'_global.png', dpi=600)
    plt.close()

    # -------------------------save layer IB-------------------------------

    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    for i in range(n_layer):
        for j in range(n_epoch):
            sns.scatterplot(x=[XT[j, i]], y=[TY[j, i]])
            plt.xlabel('I(X;T)')
            plt.ylabel('I(T;Y)')

        plt.savefig('Img/forpaper/IB/' + name + '/IB_train/' + name+'_layer_'+str(i)+'.png', dpi=600)
        plt.close()

    XT = f['rate_test'][:, :, 0]
    TY = f['rate_test'][:, :, 1]

    n_epoch = XT.shape[0]
    n_layer = XT.shape[1]

    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    for i in range(n_layer):
        for j in range(n_epoch):
            sns.scatterplot(x=[XT[j, i]], y=[TY[j, i]])
            plt.xlabel('I(X;T)')
            plt.ylabel('I(T;Y)')

    plt.savefig('Img/forpaper/IB/' + name + '/IB_test/' + name + '_global.png', dpi=600)
    plt.close()

    # -------------------------save layer IB-------------------------------

    sns.set_palette(sns.cubehelix_palette(n_epoch, start=.5, rot=-.75))
    for i in range(n_layer):
        for j in range(n_epoch):
            sns.scatterplot(x=[XT[j, i]], y=[TY[j, i]])
            plt.xlabel('I(X;T)')
            plt.ylabel('I(T;Y)')

        plt.savefig('Img/forpaper/IB/' + name + '/IB_test/' + name + '_layer_' + str(i) + '.png', dpi=600)
        plt.close()

    return True