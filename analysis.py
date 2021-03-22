from utils.rateDistortion import MultiChannelRateDistortion, RateDistortion, entropyRD, MI
from utils.DIB import reyi_entropy, calculate_MI
from utils.IDNN import entropy, mi
from utils.KDE import entropy_estimator_kl
from utils.BIN import MI_cal
import tensorflow as tf
from utils import useData
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from tqdm import tqdm

analysisPATH = './Img/analysis/'
networkPATH = './Img/network/'
analysislog = './log/analysisData/'
networklog = './log/network/'

def compareEntropy(LIST=[3,15,50,200], device='cpu', eps=0.1):

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, sharex=True, sharey=True, facecolor='white', figsize=(9, 7),
                                               dpi=600)
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = [ax1, ax2, ax3, ax4]

    count = 0
    for i in LIST:
        W = torch.ones(128, i)
        noise = torch.randn_like(W)
        sigma = 2*np.sqrt(W.shape[1]) * W.shape[0] ** (-1 / (4 + W.shape[1]))
        REDI = []
        KNN = []
        RD = []
        BIN = []
        KDE = []
        for j in tqdm(range(1, 100)):
            sample = (j / 100) * noise + (0 - j / 100) * W
            REDI.append(reyi_entropy(sample, sigma))
            KNN.append(entropy(sample.numpy()))
            RD.append(entropyRD(W=sample, device=device, eps=eps))
            BIN.append(MI_cal(sample.numpy()))
            KDE.append(entropy_estimator_kl(x=tf.convert_to_tensor(
                sample.numpy(), dtype=tf.float32), var=0.1))
        REDI = torch.tensor(REDI)
        REDI = (REDI - REDI.min()) / (REDI.max() - REDI.min()) * 10
        KNN = torch.tensor(KNN)
        KNN = (KNN - KNN.min()) / (KNN.max() - KNN.min()) * 10
        RD = torch.stack(RD)
        RD = (RD - RD.min()) / (RD.max() - RD.min()) * 10
        KDE = torch.stack(KDE)
        KDE = (KDE - KDE.min()) / (KDE.max() - KDE.min()) * 10
        BIN = torch.tensor(BIN)
        if count<2:
            BIN = (BIN - BIN.min()) / (BIN.max()-BIN.min()) * 10
        else:
            BIN = BIN / BIN.max() * 10

        ax[count].plot(REDI, 'b')
        ax[count].plot(KNN, 'y')
        ax[count].plot(RD, 'g')
        ax[count].plot(BIN, 'r')
        ax[count].plot(KDE, 'black')
        label = "n="+str(i)

        ax[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='black', markersize=5)], loc='lower right')
        count+=1
    fig.legend(['RE','KNN','CLF(ours)','BIN'], loc=(0.84,0.11))
    fig.text(0.5, 0.005, "Degree of Uncertainty", ha='center')
    fig.text(0, 0.5, "Estimated Entropy", va='center', rotation='vertical')

    plt.tight_layout()
    fig.suptitle("Entropy with various Dimension",y=1.0, fontsize='x-large')

    plt.savefig(analysisPATH+"Entropy with various Dimension.pdf", dpi=600)
    # plt.show()

    return True

def compareMI(LIST=[3,15,50,200], device='cpu', eps=0.1):

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, sharex=True, sharey=True, facecolor='white', figsize=(9, 7),
                                               dpi=600)
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = [ax1, ax2, ax3, ax4]

    count = 0
    for i in LIST:
        A = torch.randn(300, i)
        sigma = 2*np.sqrt(A.shape[1]) * A.shape[0] ** (-1 / (4 + A.shape[1]))
        REDI = []
        KNN = []
        RD = []
        BIN = []
        # generate sample & compute MI with sample and A
        for j in tqdm(range(-50, 51)):
            REDI.append(calculate_MI(x=(-j*2/100)*A, y=A, s_x=sigma, s_y=sigma))
            KNN.append(mi(x=(-j*2/100)*A.numpy(), y=A.numpy()))
            RD.append(MI(X=(-j*2/100)*A, Y=A, device=device, eps=eps))
            BIN.append(MI_cal((-j*2/100)*A.numpy()))


        REDI = torch.tensor(REDI)
        REDI = (REDI - REDI.min()) / (REDI.max() - REDI.min()) * 10
        KNN = torch.tensor(KNN)
        KNN = (KNN - KNN.min()) / (KNN.max() - KNN.min()) * 10
        RD = torch.stack(RD)
        RD = (RD - RD.min()) / (RD.max() - RD.min()) * 10
        BIN = torch.tensor(BIN)
        if count<2:
            BIN = (BIN - BIN.min()) / (BIN.max()-BIN.min()) * 10
        else:
            BIN = BIN / BIN.max() * 10

        ax[count].plot(REDI, 'b')
        ax[count].plot(KNN, 'y')
        ax[count].plot(RD, 'g')
        ax[count].plot(BIN, 'r')
        label = "n="+str(i)

        ax[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='black', markersize=5)], loc='lower right')
        count+=1
    fig.legend(['RE','KNN','CLF(ours)','BIN'], loc=(0.84,0.11))
    fig.text(0.5, 0.005, "Degree of Uncertainty", ha='center')
    fig.text(0, 0.5, "Estimated Entropy", va='center', rotation='vertical')

    plt.tight_layout()
    fig.suptitle("Comparision of MI Estimation",y=1.0, fontsize='x-large')

    plt.savefig(analysisPATH+"MI compare.pdf", dpi=600)
    # plt.show()

    return True

def computeDeEntropy(bsz, noise, mode='MNIST'):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    if mode == 'STL':
        transform = transforms.Compose([transforms.CenterCrop(size=20), transforms.ToTensor()])
    dataset, _ = useData(mode=mode, transform=transform, PATH="E:\\thesis\dataset")

    loader_org = DataLoader(
        dataset=dataset,
        batch_size=bsz,
        shuffle=True,
    )

    data = torch.zeros(10, len(noise), 5, 50)
    if mode == 'STL':
        indices = torch.LongTensor(dataset.labels)
    else:
        indices = torch.LongTensor(dataset.targets)

    for label in tqdm(range(10)):
        loader_1 = DataLoader(
            dataset=dataset,
            batch_size=bsz,
            shuffle=False,
            sampler=SubsetRandomSampler(((indices == label).nonzero().view(-1)))
        )
        for sample_1, _ in loader_1:
            for sample_org, _ in loader_org:
                sample_1 = sample_1.reshape(bsz, -1)
                sample_org = sample_org.reshape(bsz, -1)
                noi = torch.randn_like(sample_1)

                for e in range(len(noise)):
                    sample_1_noise = sample_1 + noi * noise[e]
                    sample_org_noise = sample_org + noi * noise[e]
                    sigma = 2*np.sqrt(sample_1.shape[1]) * sample_1.shape[0] ** (-1 / (4 + sample_1.shape[1]))

                    REDI = []
                    KNN = []
                    RD = []
                    BIN = []
                    KDE = []
                    for j in range(0, bsz, int(bsz/50)):
                        sample = torch.cat((sample_1_noise[:j],sample_org_noise[j:bsz]), dim=0)
                        REDI.append(reyi_entropy(sample, sigma))
                        KNN.append(entropy(sample.numpy()))
                        RD.append(entropyRD(W=sample, device='cpu', eps=0.1))
                        BIN.append(MI_cal(sample.numpy()))
                        KDE.append(entropy_estimator_kl(x=tf.convert_to_tensor(
                            sample.numpy(), dtype=tf.float32), var=0.1))
                    data[label,e,0] = torch.tensor(REDI)
                    data[label,e,1] = torch.tensor(KNN)
                    data[label,e,2] = torch.stack(RD)
                    data[label,e,3] = torch.tensor(BIN)
                    data[label,e,4] = torch.tensor(KDE)

                break
            break


    torch.save(
        obj={'data':data, 'note':['REYI','KNN','CLF(ours)','BIN']},
        f=analysislog+'degenerative'+mode+'.pth'
    )

def plotDeEntropy(name):
    f = torch.load(analysislog+name, map_location='cpu')
    data = f['data'] # R(label_num, noise_num, e_type, 50)
    note = f['note']
    data = data.permute(1, 2, 0, 3) # R(n_noise, e_type, n_label, 50)
    n_noise, e_type, n_label, n_sample = data.size()
    fig, axes = plt.subplots(1, n_noise, sharex=True, sharey=True, facecolor='white', figsize=(12, 5))

    # print((data.max(dim=3)[0] - data.min(dim=3)[0]).mean(dim=2))

    data = (data - data.min(dim=3)[0].unsqueeze(3)) / (data.max(dim=3)[0] - data.min(dim=3)[0] + 1e-10).unsqueeze(3)
    for i in range(e_type):
        data[:,i,:,:] += torch.ones_like(data[:,i,:,:])*i
    label = torch.tensor(list(range(0,n_sample))).view(1, -1).repeat(n_label, 1).view(1,-1)

    for j in range(n_noise):
        line = pd.DataFrame(torch.cat((data[j].reshape(e_type, -1), label),dim=0).T.numpy(), columns=note+['label'])
        for i in range(e_type):
            sns.lineplot(data=line, x='label', y=note[i], ax=axes[j])

    fig.legend(note)

    fig.text(0.5, 0.005, "# same kind sample", ha='center')
    fig.text(0, 0.5, "Estimated Entropy", va='center', rotation='vertical')

    plt.savefig(analysisPATH+"DeEntropy.pdf", dpi=600)

def plotIP(PLOT_LAYERS, tanhName, ReLUName, savefolder, train=True):
    if 'fnn' in tanhName:
        n_data = 3
    else:
        n_data = 4
    sns.set_style('darkgrid')

    ##
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []
    data_tanh = torch.load(networklog+tanhName, map_location='cpu')['rate_train' if train else 'rate_test']
    data_ReLU = torch.load(networklog+ReLUName, map_location='cpu')['rate_train' if train else 'rate_test']  # epoch, n_layer, k
    max_epoch = data_tanh.shape[0]
    COLORBAR_MAX_EPOCHS = max_epoch
    RT = 0
    TY = 1
    TX = 2
    if n_data == 4:
        TX = 3
    # handling data
    measures = {}
    MIY = (data_tanh[:, :, RT] - data_tanh[:, :, TY]).unsqueeze(dim=1)
    MIX = (data_tanh[:, :, RT] - data_tanh[:, :, TX]).unsqueeze(dim=1)
    MIX = MIX - MIX[0].min()
    MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'tanh': torch.cat((MIX, MIY), dim=1)})
    MIY = (data_ReLU[:, :, RT] - data_ReLU[:, :, TY]).unsqueeze(dim=1)
    MIX = (data_ReLU[:, :, RT] - data_ReLU[:, :, TX]).unsqueeze(dim=1)
    MIX = MIX - MIX[0].min()
    MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'ReLU': torch.cat((MIX, MIY), dim=1)})

    # measures in {'act': R(epoch, 2 ,n_layer)}
    # start plot

    fig = plt.figure(figsize=(10, 5))

    for actndx, (activation, vals) in enumerate(measures.items()):
        plt.subplot(1, 2, actndx + 1)
        for epoch in range(max_epoch):
            c = sm.to_rgba(epoch)
            xmvals = vals[epoch][0].numpy()[PLOT_LAYERS]
            ymvals = vals[epoch][1].numpy()[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

        # plt.ylim([0, 3.5])
        # plt.xlim([0, 14])
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()

    plt.savefig(networkPATH+savefolder+'/IP.pdf', bbox_inches='tight', dpi=600)









if __name__ == '__main__':

    compareMI([3,15,50,100], eps=0.1)
    # compareDeEntropy(bsz=500, noise=[0, 0.3, 0.6, 0.8, 1.0], mode='STL')
    # plotDeEntropy(PATH='./degenerativeData.pth')
