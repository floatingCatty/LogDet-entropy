from model import ReduNet_2D
import cupy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torchvision

def train_2D(
        model_args,
        device=0,
        n_train_sample=1000,
        n_test_sample=500,
        update_batchsize=1000,
        approximation=-1,
        top_n_acc=1,
        test_on_trainset=False
):
    '''
    model_args contains:
    e,nameta,n_class,n_channel,kernel_size,L,lr,epsilon,adversial
    '''

    redu = ReduNet_2D(
        e=model_args[0],
        nameta=model_args[1],
        n_class=model_args[2],
        in_channel=model_args[3],
        n_channel=model_args[4],
        kernel_size=model_args[5],
        L=model_args[6],
        lr=model_args[7],
        epsilon=model_args[8],
        adversial=model_args[9]
    )

    np.cuda.Device(device).use()

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if model_args[2] == 2:
        trainset_indices = ((trainset.train_labels == 0) + (trainset.train_labels == 1)).nonzero().view(-1)

        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  batch_size=n_train_sample,
                                                  shuffle=False,
                                                  sampler=SubsetRandomSampler(trainset_indices))

        testset_indices = ((testset.train_labels == 0) + (testset.train_labels == 1)).nonzero().view(-1)

        testloader = torch.utils.data.DataLoader(dataset=testset,
                                                 batch_size=n_test_sample,
                                                 shuffle=False,
                                                 sampler=SubsetRandomSampler(testset_indices))
    else:

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_train_sample,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=n_test_sample,
                                                 shuffle=True)

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

    if test_on_trainset:
        _, _, acc_train, acc_test, loss_train, loss_test = redu.estimate(
            Z=Z,
            X=Z[:,:,:,:n_test_sample],
            label_Z=label_Z,
            label_X=label_Z[:n_test_sample],
            update_batchsize=update_batchsize,
            mini_batch=approximation,
            top_n_acc=top_n_acc
        )
    else:
        _, _, acc_train, acc_test, loss_train, loss_test = redu.estimate(
            Z=Z,
            X=X,
            label_X=label_X,
            label_Z=label_Z,
            update_batchsize=update_batchsize,
            mini_batch=approximation,
            top_n_acc=top_n_acc
        )

    return acc_train, acc_test, loss_train, loss_test