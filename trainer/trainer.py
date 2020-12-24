from model import ReduNet_2D
import cupy as np
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from utils import useData
import torch

def train_2D(
        configName,

        e,
        nameta,
        n_class,
        n_channel,
        in_channel,
        kernel_size,
        L,
        lr,

        dataPATH='/home/17320015070/notespace/dataset',
        logPATH='./log',
        device=0,
        dataset='MNIST',
        n_train_sample=1000,
        n_test_sample=500,
        update_batchsize=1000,
        approximation=-1,
        top_n_acc=1
):
    '''
    model_args contains:
    e,nameta,n_class,n_channel,kernel_size,L,lr
    '''

    redu = ReduNet_2D(
        e=e,
        nameta=nameta,
        n_class=n_class,
        in_channel=in_channel,
        n_channel=n_channel,
        kernel_size=kernel_size,
        L=L,
        lr=lr
    )

    np.cuda.Device(device).use()

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset, testset = useData(mode=dataset,PATH=dataPATH, transform=transform)

    if n_class == 2:
        trainset_indices = ((trainset.train_labels == 0) + (trainset.train_labels == 1)).nonzero().view(-1)

        trainloader = DataLoader(
            dataset=trainset,
            batch_size=n_train_sample,
            shuffle=False,
            sampler=SubsetRandomSampler(trainset_indices)
        )

        testset_indices = ((testset.test_labels == 0) + (testset.test_labels == 1)).nonzero().view(-1)

        testloader = DataLoader(
            dataset=testset,
            batch_size=n_test_sample,
            shuffle=False,
            sampler=SubsetRandomSampler(testset_indices)
        )

    else:

        trainloader = DataLoader(
            trainset,
            batch_size=n_train_sample,
            shuffle=True
        )
        testloader = DataLoader(
            testset,
            batch_size=n_test_sample,
            shuffle=True
        )

    for samples in trainloader:
        Z = samples[0].permute(1, 2, 3, 0).numpy()
        label_Z = samples[1].numpy()
        Z = np.array(Z)
        label_Z = np.array(label_Z)
        break

    for samples in testloader:
        X = samples[0].permute(1, 2, 3, 0).numpy()
        X = np.array(X)
        label_X = samples[1].numpy()
        label_X = np.array(label_X)
        break

    Z, X, acc_train, acc_test, loss_train, loss_test = redu.estimate(
        Z=Z,
        X=X,
        label_X=label_X,
        label_Z=label_Z,
        update_batchsize=update_batchsize,
        mini_batch=approximation,
        top_n_acc=top_n_acc
    )

    torch.save(
        obj={
            'Z':Z,
            'X':X,
            'acc_train':acc_train,
            'acc_test':acc_test,
            'loss_train':loss_train,
            'loss_test':loss_test
        },
        f=logPATH + configName[:configName.find('.')] +'.pth'
    )


    return Z, X, acc_train, acc_test, loss_train, loss_test