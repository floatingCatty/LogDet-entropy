from utils import plotByLayer, plotByEpoch, plotList, plotMcr2TradeOff
import matplotlib.pyplot as plt
import torch
from model import ReduNet_2D, ResNet, resnet18, FNN
import cupy as cp
import numpy as np

def vv(PATH, modelType, device):
    f = torch.load(PATH, map_location='cpu')
    sample, label = f['rate_sample']

    partial = 50

    bias = torch.ones_like(sample[:partial]) + 0.03*torch.randn_like(sample[:partial])

    if modelType == 'res18':
        model = resnet18()
    elif modelType == 'fnn':
        model = FNN(gate='relu')
    else:
        raise ValueError

    model.load_state_dict(f['model_state_dict'])
    model.to(device)

    cp.cuda.Device(2).use()
    _, rate = model(sample[:partial].to(device), label[:partial], return_rate=True)
    _, bias_rate = model(bias.to(device), label[:partial], return_rate=True)
    print(np.array(rate, dtype=float).tolist())
    print(np.array(bias_rate, dtype=float).tolist())

    # plt.plot(rate)
    # plt.plot(bias_rate)
    #
    # plt.show()

    return True


if __name__ == '__main__':
    PATH = "./log/resn18_mcr2_a.pth"

    # plotList(PATH, normalize=False, yaxis=['acc_test', 'acc_train'])
    #
    # plotMcr2TradeOff(PATH, perLayer=True)

    # plotByLayer(PATH, normalize=True)
    # plotByEpoch(PATH, normalize=True)

    # vv(PATH=PATH, modelType='res18', device='cuda:2')

    a=torch.tensor([[877.5814973260215, 55.433714361580016], [1893.4091065351383, 58.06276271268597],
     [3633.7792215065565, 61.1701281982821], [814.2785317377768, 57.360818770270164],
     [254.01956715466076, 53.93461514457103]])
    b=torch.tensor([[2699.557356796505, 56.27735865168401], [2866.6194230061833, 59.41847553224734],
     [3599.8004941906847, 61.74061497310593], [765.3706878728456, 57.71472616674115],
     [242.35850360344767, 54.08697576032979]])

    plt.scatter(a[:,0], a[:,1], c='b')
    plt.scatter(b[:,0], b[:,1], c='y')

    plt.show()









