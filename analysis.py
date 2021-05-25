from utils.DIB import reyi_entropy
from utils.IDNN import entropy
from utils.KDE import entropy_estimator_kl
from utils.BIN import MI_cal
from utils import useData
from model import autoencoder
from utils.tsne import tsne
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from utils.LogDet import mi as mi_rd
from utils.LogDet import LogDet
from utils.utils import generateNormalizedClusteredData as gnd


analysisPATH = './Img/analysis/'
networkPATH = './Img/network/'
analysislog = './log/analysisData/'
networklog = './log/network/'

def plotNormalization():
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(6, 2.5), dpi=600)

    rate_relu = []
    rate_tanh = []
    b_list = [10, 100, 500, 1000]
    for n in range(len(b_list)):
        w = 1+torch.randn(b_list[n], b_list[n])
        a = torch.randn(3000, b_list[n])

        for i in tqdm(range(100)):
            # sigma = 2*np.sqrt(a.shape[1]) * a.shape[0] ** (-1 / (4 + a.shape[1]))

            b =  a.matmul((i / 10.)*w).double()
            rate_relu.append(mi_rd(a, torch.relu(b), device='cpu', mode='cov'))
            rate_tanh.append(mi_rd(a, torch.tanh(b), device='cpu', mode='cov'))
            # append(calculate_MI(a, torch.relu(b), sigma, sigma))
            # rate_tanh.append(calculate_MI(a, torch.tanh(b), sigma, sigma))

        axes[1][n].plot(rate_relu)
        axes[0][n].plot(rate_tanh)
        axes[1][n].set_xlabel('d='+str(b_list[n]), size=7)
        axes[0][n].tick_params(axis='both', which='major', labelsize=5)
        axes[1][n].tick_params(axis='both', which='major', labelsize=5)
        rate_relu = []
        rate_tanh =[]
    # axes[1][0].set_ylabel('LogDet', size=7)
    # axes[0][0].set_ylabel('α-Renyi', size=7)
    axes[1][0].set_ylabel('ReLU', size=7)
    axes[0][0].set_ylabel('tanh', size=7)

    # axes[1][0].set_ylim(10,15)
    # axes[1][1].set_ylim(10, 15)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.15)
    plt.xticks([0,20,40,60,80,100],[0,2,4,6,8,10])
    fig.text(0.52, 0.02, "w", ha='center')
    fig.text(0, 0.5, "I(X;T)", va='center', rotation='vertical')

    plt.savefig(analysisPATH+"actcompare.pdf", dpi=600)

def compareEntropy(LIST=[3,15,50,200], device='cpu'):
    sns.set_style('whitegrid')
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(9, 7), dpi=600)

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = [ax1, ax2, ax3, ax4]
    note = ['REYI','KNN','LD(ours)','KDE','BIN']
    count = 0
    for i in LIST:
        W = torch.ones(128, i).to(device)
        noise = torch.randn_like(W).to(device)
        sigma = 2*np.sqrt(W.shape[1]) * W.shape[0] ** (-1 / (4 + W.shape[1]))
        data = [[],[],[],[],[]]
        for j in tqdm(range(1, 100)):
            sample = (j / 100) * noise + (1 - j / 100) * W
            data[0].append(reyi_entropy(sample, sigma))
            data[1].append(entropy(sample.cpu().numpy()))
            data[2].append(LogDet(A=sample, device=device, mode='cov', beta=1))
            data[3].append(entropy_estimator_kl(x=sample, var=0.1))
            data[4].append(MI_cal(sample.clone().cpu().numpy()))
        data = torch.tensor(data)
        data = (data - data.min(dim=1)[0].unsqueeze(1)) / (data.max(dim=1)[0] - data.min(dim=1)[0]).unsqueeze(1)
        data = data * 10
        label = torch.tensor(list(range(1,100))) / 100
        data = data.cpu().numpy()
        label = label.cpu().numpy()

        for q in range(len(note)):
            sns.lineplot(x=label,y=data[q], ax=ax[count])

        label = "d="+str(i)

        legend = ax[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='black', markersize=5)], loc='lower right')
        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('white')

        ax[count].xlabel = None
        ax[count].ylabel = None
        count+=1
    legend = fig.legend(note, loc=(0.858,0.58))
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('white')

    fig.text(0.52, 0.007, "Degree of Uncertainty", ha='center')
    fig.text(0.007, 0.54, "Estimated Entropy", va='center', rotation='vertical')

    plt.tight_layout()

    plt.savefig(analysisPATH+"Entropy with various Dimension.pdf", dpi=600)
    # plt.show()

    return True

def TheoreticalEntropy(LIST=[3,15,50,200], device='cpu'):
    sns.set_style('whitegrid')
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(9, 7), dpi=600)

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = [ax1, ax2, ax3, ax4]
    note = ['REYI','KNN','LD(ours)','KDE','BIN']
    count = 0
    for i in LIST:
        data = [[],[],[],[],[]]
        mean = [0] * i
        for j in tqdm(range(0, 100)):
            cov = torch.ones(i,i)*(1-j/100)+(j/100)*torch.eye(i)
            sample = np.random.multivariate_normal(mean=mean, cov=cov.cpu().numpy(), size=128)
            sample = torch.tensor(sample).to(device)
            sigma = 2 * np.sqrt(sample.shape[1]) * sample.shape[0] ** (-1 / (4 + sample.shape[1]))
            data[0].append(reyi_entropy(sample, sigma))
            data[1].append(entropy(sample.cpu().numpy()))
            data[2].append(LogDet(sample, mode='pearson', device=device))
            data[3].append(entropy_estimator_kl(x=sample, var=0.1))
            data[4].append(MI_cal(sample.clone().cpu().numpy()))
        data = torch.tensor(data)
        data = (data - data.min(dim=1)[0].unsqueeze(1)) / (data.max(dim=1)[0] - data.min(dim=1)[0]).unsqueeze(1)
        data = data * 10
        label = torch.tensor(list(range(0,100))) / 100
        data = data.cpu().numpy()
        label = label.cpu().numpy()

        for q in range(len(note)):
            sns.lineplot(x=label,y=data[q], ax=ax[count])

        label = "d="+str(i)

        legend = ax[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='black', markersize=5)], loc='lower right')
        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('white')

        ax[count].xlabel = None
        ax[count].ylabel = None
        count+=1
    legend = fig.legend(note, loc=(0.858,0.58))
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('white')

    fig.text(0.52, 0.007, "Correlation Factor", ha='center')
    fig.text(0.007, 0.54, "Estimated Entropy", va='center', rotation='vertical')

    plt.tight_layout()

    plt.savefig(analysisPATH+"Theoretical Entropy.pdf", dpi=600)
    # plt.show()

    return True

def compareMI(LIST=[3,15,50], sample=128):
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    fig, axes = plt.subplots(1,3, sharex=True, facecolor='white', figsize=(6, 2),
                                               dpi=600)
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ylim = [
        [0,0.5],
        [5,15],
        [90,97.5]
    ]
    count = 0
    for i in LIST:
        RD = []
        # generate sample & compute MI with sample and A
        correlations = np.linspace(-0.9, 0.9, 19)

        mean = [0]*2*i
        cov = torch.eye(2*i)
        for rho in tqdm(correlations):
            cov[0:i,i:2*i] = torch.eye(i)*rho
            cov[i:2*i,0:i] = torch.eye(i)*rho

            rho_data = np.random.multivariate_normal(mean=mean, cov=cov.numpy(), size=sample)
            rho_data = torch.tensor(rho_data)
            x = rho_data[:, 0:i]
            y = rho_data[:, i:2*i]

            RD.append(mi_rd(x, y, device='cpu', mode='pearson', beta=1))
            # bin in the last term

        axes[count].plot(correlations,RD,c='mediumblue')
        axes[count].plot(correlations,RD, 'o',c='mediumblue')
        # axes[count].set_ylim([min(RD) - 5, max(RD) + 5])
        axes[count].set_ylim(ylim[count])
        axes[count].set_xlim([-1, 1])
        # ax[count].set_xticklabels([str(j) for j in correlations], fontsize='small')
        label = "d="+str(i)

        axes[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='black', markersize=4)], loc='upper right')
        count+=1

    fig.text(0.5, 0.05, 'ρ', ha='center')
    fig.text(0, 0.5, "I(X;Y)", va='center', rotation='vertical')

    plt.tight_layout()

    plt.savefig(analysisPATH+"MIviaD.pdf", dpi=600)

    return True

def MIviaN(LIST=[128,256,512]):
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    fig, axes = plt.subplots(1,3, sharex=True, facecolor='white', figsize=(6, 2),
                                               dpi=600)
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']

    count = 0
    for i in LIST:
        RD = []
        # generate sample & compute MI with sample and A
        correlations = np.linspace(-0.9, 0.9, 19)
        mean = [0]*2000
        cov = torch.eye(2000)
        for rho in tqdm(correlations):
            cov[0:1000,1000:2000] = torch.eye(1000)*rho
            cov[1000:2000,0:1000] = torch.eye(1000)*rho

            rho_data = np.random.multivariate_normal(mean=mean, cov=cov.numpy(), size=i)
            rho_data = torch.tensor(rho_data)
            x = rho_data[:, 0:1000]
            y = rho_data[:, 1000:2000]

            RD.append(mi_rd(x, y, device='cpu', mode='pearson'))
            # bin in the last term

        axes[count].plot(correlations,RD, c='mediumblue')
        axes[count].plot(correlations,RD, 'o', c='mediumblue')
        axes[count].set_title('#sample='+str(i))
        axes[count].set_ylim([min(RD)-5, max(RD)+5])
        axes[count].set_xlim([-1, 1])
        # ax[count].set_xticklabels([str(j) for j in correlations], fontsize='small')
        label = "#sample="+str(i)

        # axes[count].legend(handles=[Line2D([0], [0], marker='o', color='w', label=label,
        #                   markerfacecolor='black', markersize=5)], loc='upper right')
        count+=1

    fig.text(0.5, 0.05, 'ρ', ha='center')
    fig.text(0, 0.5, "I(X;Y)", va='center', rotation='vertical')

    plt.tight_layout()

    plt.savefig(analysisPATH+"MIviaN.pdf", dpi=600)

    return True

def computeDeEntropy(bsz, noise, mode='MNIST'):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    # if mode == 'STL':
    #     transform = transforms.Compose([transforms.CenterCrop(size=25), transforms.ToTensor()])
    dataset, _ = useData(mode=mode, transform=transform, PATH="../dataset")

    loader_org = DataLoader(
        dataset=dataset,
        batch_size=bsz,
        shuffle=True,
    )

    data = torch.zeros(10, len(noise), 1, 50)
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
                sample_1 = sample_1.reshape(bsz, -1).to('cuda')
                sample_org = sample_org.reshape(bsz, -1).to('cuda')
                noi = torch.randn_like(sample_1).to('cuda')

                for e in tqdm(range(len(noise))):
                    sample_1_noise = sample_1 + noi * noise[e]
                    sample_org_noise = sample_org + noi * noise[e]
                    sigma = 2*np.sqrt(sample_1.shape[1]) * sample_1.shape[0] ** (-1 / (4 + sample_1.shape[1]))

                    # REDI = []
                    # KNN = []
                    RD = []
                    # BIN = []
                    # KDE = []
                    for j in range(0, bsz, int(bsz/50)):
                        sample = torch.cat((sample_1_noise[:j],sample_org_noise[j:bsz]), dim=0)
                        # REDI.append(reyi_entropy(sample, sigma))
                        # KNN.append(entropy(sample.numpy()))
                        RD.append(LogDet(A=sample, device='cuda', mode='cov'))
                        #
                        # KDE.append(entropy_estimator_kl(x=sample, var=0.1))
                        # BIN.append(MI_cal(sample.numpy()))

                    # print(RD)
                    # data[label,e,0] = torch.tensor(REDI)
                    # data[label,e,1] = torch.tensor(KNN)
                    data[label,e,0] = torch.stack(RD)
                    # data[label,e,3] = torch.tensor(BIN)
                    # data[label,e,4] = torch.tensor(KDE)


                break
            break


    torch.save(
        obj={'data':data, 'note':['REYI','KNN','CLF(ours)','BIN', 'KDE']},
        f=analysislog+'degenerative_LogDet'+mode+'.pth'
    )

def plotDeEntropy(namelist):
    import matplotlib


    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True, facecolor='white', figsize=(12, 12))
    ax = 0
    noise = ['0', '0.2', '0.6', '0.8', '1.0']
    dataName = ['MNIST', 'STL', 'CIFAR10']
    for name in namelist:
        f = torch.load(analysislog+name, map_location='cpu')
        data = f['data'] # R(label_num, noise_num, e_type, 50)
        note = f['note']
        note[2] = 'LD(ours)'
        data = data.permute(1, 2, 0, 3) # R(n_noise, e_type, n_label, 50)
        n_noise, e_type, n_label, n_sample = data.size()


        # print((data.max(dim=3)[0] - data.min(dim=3)[0]).mean(dim=2))

        data = (data - data.min(dim=3)[0].unsqueeze(3)) / (data.max(dim=3)[0] - data.min(dim=3)[0] + 1e-10).unsqueeze(3)
        for i in range(e_type):
            data[:,i,:,:] += torch.ones_like(data[:,i,:,:])*i
        label = torch.tensor(list(range(0,n_sample))).view(1, -1).repeat(n_label, 1).view(1,-1) * 2

        for j in range(n_noise):
            line = pd.DataFrame(torch.cat((data[j].reshape(e_type, -1), label),dim=0).T.numpy(), columns=note+['label'])
            for i in range(e_type):
                sns.lineplot(data=line, x='label', y=note[i], ax=axes[ax][j])
            axes[ax][j].set_xlabel('σ = '+ noise[j])
            axes[ax][j].set_ylabel(dataName[ax])
        ax += 1

    fig.text(0.52, 0.005, "percent of same kind samples", ha='center')
    fig.text(0, 0.5, "Estimated Entropy", va='center', rotation='vertical')
    legend = fig.legend(note, ncol=5, loc='upper right')
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('white')
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    plt.tight_layout()

    plt.savefig(analysisPATH+"DeEntropy_all.pdf", dpi=600)

def plotIP(tanhName, ReLUName, savefolder, train=True):
    import matplotlib

    # matplotlib.rc('xtick', labelsize=5)
    # matplotlib.rc('ytick', labelsize=5)

    if 'fnn' in tanhName:
        n_data = 4
    else:
        n_data = 5
    sns.set_style('darkgrid')
    tanh = torch.load(networklog+tanhName, map_location='cpu')
    ReLU = torch.load(networklog+ReLUName, map_location='cpu')
    ##

    data_tanh = tanh['rate_train' if train else 'rate_test']
    data_ReLU = ReLU['rate_train' if train else 'rate_test']  # epoch, n_layer, k
    n_layer = data_ReLU.shape[1]
    max_epoch = data_tanh.shape[0]
    # max_epoch = 200
    COLORBAR_MAX_EPOCHS = max_epoch
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []
    IXT = 0
    ITY = 1
    # handling data
    measures = {}
    MIY = (data_tanh[:, :, 6]+data_tanh[:, :, 7]-data_tanh[:, :, 9]).unsqueeze(dim=1)
    MIX = (data_tanh[:, :, 5]+data_tanh[:, :, 6]-data_tanh[:, :, 8]).unsqueeze(dim=1)
    # MIX = MIX - MIX[0].min()
    # MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    # MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'tanh': torch.cat((MIX, MIY), dim=1)})
    MIY = (data_ReLU[:, :, 6]+data_ReLU[:, :, 7]-data_ReLU[:, :, 9]).unsqueeze(dim=1)
    MIX = (data_ReLU[:, :, 5]+data_ReLU[:, :, 6]-data_ReLU[:, :, 8]).unsqueeze(dim=1)
    # MIX = MIX - MIX[0].min()
    # MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    # MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'ReLU': torch.cat((MIX, MIY), dim=1)})

    # measures in {'act': R(epoch, 2 ,n_layer)}
    # start plot

    fig, axes = plt.subplots(2, n_layer, sharex=False, sharey=False, figsize=(1.5*n_layer, 4))

    for i in range(n_layer):
        for actndx, (activation, vals) in enumerate(measures.items()):
            for epoch in tqdm(range(0,max_epoch,5)):
                c = sm.to_rgba(epoch)
                xmvals = vals[epoch][0].numpy()[[i]]
                ymvals = vals[epoch][1].numpy()[[i]]

                # plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
                axes[actndx][i].scatter(xmvals, ymvals, s=20, facecolors=c, edgecolor='none', zorder=2)
            axes[actndx][i].set_ylim([vals[:,1,i].min(), vals[:,1,i].max()])
            axes[actndx][i].tick_params(axis='both', which='major', labelsize=8)
            axes[actndx][i].tick_params(axis='both', which='major', labelsize=8)
        # plt.xscale('symlog')
        # plt.ylim([0, 3.5])
        # plt.xlim([0, 14])
        axes[0][i].set_title('layer '+str(i+1), size=8)


    axes[0][0].set_ylabel('tanh')
    axes[1][0].set_ylabel('ReLU')
    fig.text(0.52, 0, "I(X;T)", ha='center', fontsize='large')
    fig.text(0, 0.5, "I(T;Y)", va='center', rotation='vertical', fontsize='large')

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)

    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    if train:
        plt.savefig(networkPATH+savefolder+'/IP_train.pdf', bbox_inches='tight', dpi=600)
    else:
        plt.savefig(networkPATH + savefolder + '/IP_test.pdf', bbox_inches='tight', dpi=600)
def plotCL(tanhName, ReLUName, savefolder, train=True):
    import matplotlib

    # matplotlib.rc('xtick', labelsize=5)
    # matplotlib.rc('ytick', labelsize=5)

    sns.set_style('darkgrid')
    tanh = torch.load(networklog+tanhName, map_location='cpu')
    ReLU = torch.load(networklog+ReLUName, map_location='cpu')

    data_tanh = tanh['rate_train' if train else 'rate_test']
    data_ReLU = ReLU['rate_train' if train else 'rate_test']  # epoch, n_layer, k
    c_tanh = tanh['channel_train' if train else 'channel_test']
    c_ReLU = ReLU['channel_train' if train else 'channel_test']

    n_layer = data_ReLU.shape[1]
    max_epoch = data_tanh.shape[0]
    # max_epoch = 200
    COLORBAR_MAX_EPOCHS = max_epoch
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []
    IXT = 0
    ITY = 1
    # handling data
    measures = {}
    MIY = (data_tanh[:, :, 6] + data_tanh[:, :, 7] - data_tanh[:, :, 9]).unsqueeze(dim=1)
    MIX = c_tanh.unsqueeze(dim=1)
    # MIX = MIX - MIX[0].min()
    # MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    # MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'tanh': torch.cat((MIX, MIY), dim=1)})
    MIY = (data_ReLU[:, :, 6] + data_ReLU[:, :, 7] - data_ReLU[:, :, 9]).unsqueeze(dim=1)
    MIX = c_ReLU.unsqueeze(dim=1)
    # MIX = MIX - MIX[0].min()
    # MIX = MIX / MIX.mean(dim=0).unsqueeze(dim=0)
    # MIY = MIY / MIY.mean(dim=0).unsqueeze(dim=0)
    measures.update({'ReLU': torch.cat((MIX, MIY), dim=1)})

    # measures in {'act': R(epoch, 2 ,n_layer)}
    # start plot

    fig, axes = plt.subplots(2, n_layer, sharex=False, sharey=False, figsize=(1.5*n_layer, 4))

    for i in range(n_layer):
        for actndx, (activation, vals) in enumerate(measures.items()):
            for epoch in tqdm(range(0,max_epoch,5)):
                c = sm.to_rgba(epoch)
                xmvals = vals[epoch][0].numpy()[[i]]
                ymvals = vals[epoch][1].numpy()[[i]]

                # plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
                axes[actndx][i].scatter(xmvals, ymvals, s=20, facecolors=c, edgecolor='none', zorder=2)
            axes[actndx][i].set_ylim([vals[:,1,i].min(), vals[:,1,i].max()])
            axes[actndx][i].tick_params(axis='both', which='major', labelsize=8)
            axes[actndx][i].tick_params(axis='both', which='major', labelsize=8)
        # plt.xscale('symlog')
        # plt.ylim([0, 3.5])
        # plt.xlim([0, 14])
        axes[0][i].set_title('layer '+str(i+1), size=8)


    axes[0][0].set_ylabel('tanh')
    axes[1][0].set_ylabel('ReLU')
    fig.text(0.52, 0, "LTC(L)", ha='center', fontsize='large')
    fig.text(0, 0.5, "I(T;Y)", va='center', rotation='vertical', fontsize='large')

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)

    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    if train:
        plt.savefig(networkPATH+savefolder+'/CL_train.pdf', bbox_inches='tight', dpi=600)
    else:
        plt.savefig(networkPATH + savefolder + '/CL_test.pdf', bbox_inches='tight', dpi=600)


def relationHI(tanhName, ReLUName, savefolder, train=True):
    tanh = torch.load(networklog + tanhName, map_location='cpu')
    ReLU = torch.load(networklog + ReLUName, map_location='cpu')
    ##
    data_tanh = tanh['rate_train' if train else 'rate_test']
    data_ReLU = ReLU['rate_train' if train else 'rate_test']# epoch, n_layer, k

    # normalize


    k = data_tanh.shape[2]
    max_epoch = data_tanh.shape[0]
    n_layer = data_ReLU.shape[1]
    fig, axes = plt.subplots(n_layer, 2, sharex=True, sharey=False, figsize=(15, 1.25 * n_layer))

    axes[0][0].set_title('tanh')
    axes[0][1].set_title('ReLU')
    for i in range(n_layer):
        axes[i][0].plot(data_tanh[:, i, -2].numpy())
        axt = axes[i][0].twinx()
        axt.plot(data_tanh[:,i,0].numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

        axes[i][1].plot(data_ReLU[:, i, -2].numpy())
        axt = axes[i][1].twinx()
        axt.plot(data_ReLU[:, i, 0].numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

    custom_lines = [Line2D([0], [0], lw=2.5),
                    Line2D([0], [0], color='coral', lw=2.5)]
    plt.legend(custom_lines, ['I(X;T)','H(T)'])
    plt.tight_layout()
    plt.savefig(networkPATH + savefolder + '/HI_compare.pdf', bbox_inches='tight', dpi=600)

def GeometricX(tanhName, ReLUName, savefolder, train=True):
    tanh = torch.load(networklog + tanhName, map_location='cpu')
    ReLU = torch.load(networklog + ReLUName, map_location='cpu')
    ##
    data_tanh = tanh['rate_train' if train else 'rate_test']
    data_ReLU = ReLU['rate_train' if train else 'rate_test']# epoch, n_layer, k

    # normalize


    k = data_tanh.shape[2]
    max_epoch = data_tanh.shape[0]
    n_layer = data_ReLU.shape[1]
    fig, axes = plt.subplots(n_layer, 2, sharex=True, sharey=False, figsize=(8, 0.83 * n_layer))

    axes[0][0].set_title('tanh')
    axes[0][1].set_title('ReLU')
    for i in range(n_layer):
        axes[i][0].plot(data_tanh[:, i, -2].numpy())
        axt = axes[i][0].twinx()
        axt.plot((data_tanh[:,i,0]-data_tanh[:,i,1]).numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

        axes[i][1].plot(data_ReLU[:, i, -2].numpy())
        axt = axes[i][1].twinx()
        axt.plot((data_ReLU[:,i,0]-data_ReLU[:,i,1]).numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

    custom_lines = [Line2D([0], [0], lw=2.5),
                    Line2D([0], [0], color='coral', lw=2.5)]
    fig.legend(custom_lines, ['I(X;T)','Sc(T)'], loc='lower right')
    fig.text(0.52, 0.005, "Epoch", ha='center', fontsize='large')
    fig.text(0, 0.5, "I(X;T)", va='center', rotation='vertical', fontsize='large')
    fig.text(1, 0.5, "Sc(T)", va='center', rotation='vertical', fontsize='large')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(networkPATH + savefolder + '/GeometricX.pdf', bbox_inches='tight', dpi=600)

def GeometricY(tanhName, ReLUName, savefolder, train=True):
    tanh = torch.load(networklog + tanhName, map_location='cpu')
    ReLU = torch.load(networklog + ReLUName, map_location='cpu')
    ##
    data_tanh = tanh['rate_train' if train else 'rate_test']
    data_ReLU = ReLU['rate_train' if train else 'rate_test']# epoch, n_layer, k

    # normalize


    k = data_tanh.shape[2]
    max_epoch = data_tanh.shape[0]
    n_layer = data_ReLU.shape[1]
    fig, axes = plt.subplots(n_layer, 2, sharex=True, sharey=False, figsize=(8, 0.83 * n_layer))

    axes[0][0].set_title('tanh')
    axes[0][1].set_title('ReLU')
    for i in range(n_layer):
        axes[i][0].plot(data_tanh[:, i, -1].numpy())
        axt = axes[i][0].twinx()
        axt.plot((data_tanh[:,i,0]).numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

        axes[i][1].plot(data_ReLU[:, i, -1].numpy())
        axt = axes[i][1].twinx()
        axt.plot((data_ReLU[:, i, 0]).numpy(), c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

    custom_lines = [Line2D([0], [0], lw=2.5),
                    Line2D([0], [0], color='coral', lw=2.5)]
    fig.legend(custom_lines, ['I(T;Y)','Sg(T)'], loc='lower right')
    fig.text(0.52, 0.005, "Epoch", ha='center', fontsize='large')
    fig.text(0, 0.5, "I(T;Y)", va='center', rotation='vertical', fontsize='large')
    fig.text(1, 0.5, "Sc(T)", va='center', rotation='vertical', fontsize='large')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    # plt.savefig(networkPATH + savefolder + '/GeometricY.pdf', bbox_inches='tight', dpi=600)

def LossMI(tanhName, ReLUName, savefolder, train=True):
    tanh = torch.load(networklog + tanhName, map_location='cpu')
    ReLU = torch.load(networklog + ReLUName, map_location='cpu')
    ##
    data_tanh_test = tanh['rate_test']
    data_ReLU_test = ReLU['rate_test']# epoch, n_layer, k
    data_tanh_train = tanh['rate_train']
    data_ReLU_train = ReLU['rate_train']  # epoch, n_layer, k
    acc_tanh_train = tanh['acc_train']
    loss_tanh_train = tanh['loss_train']
    acc_ReLU_train = ReLU['acc_train']
    loss_ReLU_train = ReLU['loss_train']

    acc_tanh_test = tanh['acc_test']
    loss_tanh_test = tanh['loss_test']
    acc_ReLU_test = ReLU['acc_test']
    loss_ReLU_test = ReLU['loss_test']
    # normalize


    n_layer = data_ReLU_test.shape[1]
    fig, axes = plt.subplots(n_layer, 2, sharex=True, sharey=False, figsize=(8, 0.83 * n_layer))

    axes[0][0].set_title('tanh')
    axes[0][1].set_title('ReLU')
    for i in range(n_layer):
        axes[i][0].plot(data_tanh_train[:, i, -2].numpy())
        axt = axes[i][0].twinx()
        axt.plot(acc_tanh_test, c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

        axes[i][1].plot(data_ReLU_train[:, i, -2].numpy())
        axt = axes[i][1].twinx()
        axt.plot(acc_ReLU_test, c='coral', linestyle='--')
        axt.spines['right'].set_color('coral')

    custom_lines = [Line2D([0], [0], lw=2.5),
                    Line2D([0], [0], color='coral', lw=2.5)]
    fig.legend(custom_lines, ['I(X;T)','loss'], loc='lower right')
    fig.text(0.52, 0.005, "Epoch", ha='center', fontsize='large')
    fig.text(0, 0.5, "I(T;Y)", va='center', rotation='vertical', fontsize='large')
    fig.text(1, 0.5, "Sc(T)", va='center', rotation='vertical', fontsize='large')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(networkPATH + savefolder + '/LossMI.pdf', bbox_inches='tight', dpi=600)

def computePearson(x,y):
    x = x - x.mean(dim=1).unsqueeze(1)
    y = y - y.mean(dim=1).unsqueeze(1)
    covx = torch.sqrt((x**2).sum(dim=1))
    covy = torch.sqrt((y**2).sum(dim=1))
    coxy = (x*y).sum(dim=1)

    return coxy / (covx*covy)

def LayerCapacity(file_list):
    from matplotlib import cm
    n = len(file_list)
    # fig = plt.twinx()

    fig, axes = plt.subplots(2, n, sharex=True, sharey=False, figsize=(4*n, 8))
    for i in range(n):
        f = torch.load(networklog + file_list[i], map_location='cpu')
        channel = f['channel_train']
        MI = f['rate_train']
        n_layer = channel.shape[1]
        c = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_layer))
        c._A = []
        for j in range(1, n_layer):
            mi = MI[:, j, 0]
            axes[0][i].plot(mi - mi.min(), c=c.to_rgba(j))
            axes[1][i].plot(channel[:,j]-channel[:,j].min(), c=c.to_rgba(j))

    axes[0][0].set_ylabel('I(X;T)')
    axes[1][0].set_ylabel('C(L)')

    fig.legend(['1', '2', '3', '4', '5'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(analysisPATH+'LC.pdf', dpi=600)

    return True

def plotAcc(tanhName,ReLUName,savefolder,train):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    if train:
        f = torch.load(networklog + tanhName, map_location='cpu')
        acc_tanh = f['acc_train']
        f = torch.load(networklog + ReLUName, map_location='cpu')
        acc_ReLU = f['acc_train']
        outname = "/acc_train.pdf"
    else:
        f = torch.load(networklog + tanhName, map_location='cpu')
        acc_tanh = f['acc_test']
        f = torch.load(networklog + ReLUName, map_location='cpu')
        acc_ReLU = f['acc_test']
        outname = "/acc_test.pdf"

    plt.plot(acc_tanh)
    plt.plot(acc_ReLU)
    plt.legend(['tanh','ReLU'])


    plt.savefig(networkPATH+savefolder+outname, dpi=600)


def atec(in_dim, out_dim, sample_num, n_class, epoch, lr=0.1, eps=0.05, device='cuda'):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    data = gnd(cluster=torch.randn(n_class, in_dim), std=1, num=sample_num // n_class).to(device)
    target = torch.LongTensor(list(range(n_class)))
    target = target.reshape(-1,1).repeat(1,sample_num // n_class).reshape(-1)
    ae = autoencoder(eps)
    T = torch.randn(sample_num, out_dim).to(device)
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
    with torch.no_grad():
        data_plot = tsne(data, 2, 50, 20.0)

    axes[0].scatter(data_plot[:, 0].cpu(), data_plot[:, 1].cpu(), 20, target.cpu())
    axes[0].set_xlabel('X')

    with torch.no_grad():
        T_plot = tsne(T, 2, 50, 20.0)
    axes[1].scatter(T_plot[:, 0].cpu(), T_plot[:, 1].cpu(), 20, target.cpu())
    axes[1].set_xlabel('original T')

    for i in range(2,5):
        T = ae.compute(
            X=data,
            T=T,
            epoch=epoch // 3,
            lr=lr,
            device='cuda'
        )
        with torch.no_grad():
            T_plot = tsne(T, 2, 50, 20.0)
        axes[i].scatter(T_plot[:, 0].cpu(), T_plot[:, 1].cpu(), 20, target.cpu())
        axes[i].set_xlabel('T after '+str((i-1)*(epoch//3))+' epochs')
    plt.tight_layout()
    plt.savefig(analysisPATH+'autoencoder_gaussian.pdf', dpi=600)

    return True


if __name__ == '__main__':
    import sys
    import os
    os.chdir(sys.path[0])
    compareEntropy(device='cuda')
    # computeMI(bsz=2000, mode='CIFAR10')
    # plotRealMI("MIrealCIFAR10.pth")
    # compareMI([3, 100, 1000],sample=128)
    # MIviaN()
    # TheoreticalEntropy()
    # computeDeEntropy(bsz=500, noise=[0, 0.3, 0.6, 0.8, 1.0], mode='STL')
    # namelist = ['degenerativeMNIST.pth','degenerativeSTL.pth','degenerativeCIFAR10.pth']
    # plotDeEntropy(namelist)
    # plotIP(
    #     tanhName="fnn/7l/fnn_tanh.pth",
    #     ReLUName="fnn/7l/fnn_ReLU.pth",
    #     savefolder="fnn/7l/",
    #     train=False
    # )
    # plotCL(
    #     tanhName="fnn/7l/fnn_tanh.pth",
    #     ReLUName="fnn/7l/fnn_ReLU.pth",
    #     savefolder="fnn/7l/",
    #     train=False
    # )
    # plotAcc(
    #     tanhName="fnn/fixed 1/fnn_tanh.pth",
    #     ReLUName="fnn/fixed 1/fnn_ReLU.pth",
    #     savefolder="fnn/fixed 1/",
    #     train=False
    # )

    # plotNormalization()
    # relationHI(
    #     tanhName="CaL/CIFAR10/CaL38_SGD_tanh_e.pth",
    #     ReLUName="CaL/CIFAR10/CaL38_SGD_ReLU_e.pth",
    #     savefolder="CAL/CIFAR10/",
    #     train=True
    # )
    # GeometricX(
    #     tanhName="CaL/CIFAR10/CaL38_SGD_tanh_bn.pth",
    #     ReLUName="CaL/CIFAR10/CaL38_SGD_ReLU_bn.pth",
    #     savefolder="CaL/CIFAR10/bn",
    #     train=False
    # )
    # GeometricY(
    #     tanhName="fnn/MNIST/fnn_tanh_e.pth",
    #     ReLUName="fnn/MNIST/fnn_ReLU_e.pth",
    #     savefolder="fnn/bn/",
    #     train=False
    # )
    # tanh = torch.load(networklog + "CaL/CIFAR10/CaL38_SGD_tanh_e.pth", map_location='cpu')['rate_train']
    # ReLU = torch.load(networklog + "CaL/CIFAR10/CaL38_SGD_ReLU_e.pth", map_location='cpu')['rate_train']

    # LossMI(
    #     tanhName="fnn/CIFAR10/fnn_ReLU.pth",
    #     ReLUName="fnn/CIFAR10/fnn_ReLU.pth",
    #     savefolder="fnn/",
    #     train=False
    # )

    # tanh = tanh[:,:,[0,-2]].permute(2,1,0)
    # ReLU = ReLU[:, :, [0, -2]].permute(2,1,0)
    #
    # print(computePearson(x=tanh[0], y=tanh[1]))
    # print(computePearson(x=ReLU[0], y=ReLU[1]))
    #
    # LayerCapacity(
    #     file_list=[
    #         "/fnn/CIFAR10/fnn_ReLU.pth",
    #         "/fnn/CIFAR10/fnn_tanh.pth"
    #     ]
    # )

    # atec(
    #     in_dim=200,
    #     out_dim=20,
    #     sample_num=500,
    #     n_class=10,
    #     epoch=600,
    #     lr=0.1,
    #     device='cuda'
    # )