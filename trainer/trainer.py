from model import ReduNet_2D, ResNet, resnet18, FNN
import cupy as np
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader, random_split
from utils import useData, top_n, MaximalCodingRateReduction
import torch.nn.functional as F
import torchvision.models as models
import torch
from tqdm import tqdm

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
        label_proportion=-1,
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
        label_proportion=label_proportion,
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
        f=logPATH+ '/' + configName[:configName.find('.')] +'.pth'
    )


    return Z, X, acc_train, acc_test, loss_train, loss_test

def train_IC(
        configName,
        model_parameters,
        modelType,
        rateSize,
        lr,
        momentum,
        n_class,
        epochSize,
        n_train_sample,
        n_test_sample,
        device,
        Rdevice,
        dataset,
        dataPATH,
        logPATH,
        top_n_acc,
):
    '''
    modelType: model supported in torchvision
    '''

    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []
    rate = {}

    if modelType == 'res18':
        model = resnet18()
    elif modelType == 'fnn':
        model = FNN(gate=model_parameters['gate'])
    elif modelType == 'vgg':
        model = models.vgg11(pretrained=False, progress=True, num_classes=n_class)
    else:
        raise ValueError

    model = model.train().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    trainset, testset = useData(mode=dataset, PATH=dataPATH, transform=transform)

    rate_X = torch.stack([trainset[i][0] for i in range(rateSize)])
    rate_Y = torch.tensor([trainset[i][1] for i in range(rateSize)])

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


    # N, C, H, W = train_data.size()
    # n, _, _, _ = test_data.size()
    # if C == 1:
    #     train_data = train_data.expand(N, 3, H, W)
    #     test_data = test_data.expand(n, 3, H, W)



    def cross_validation_epoch(
            model,
            loader,
            optimizer,
            top_n_acc
    ):
        loss = 0
        acc = 0

        n_iter = len(loader)

        for data, label in tqdm(loader):
            pre = model(data.to(device))
            optimizer.zero_grad()

            l = F.cross_entropy(target=label.to(device), input=pre)

            pre = pre.detach().cpu().numpy()
            ac = top_n(pre=pre, label=label.cpu().numpy(), n=top_n_acc)

            l.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            loss += l.item()
            acc += ac

        return loss / n_iter, acc / n_iter

    def estimate(
            model,
            loader,
            top_n_acc
    ):
        loss = 0
        acc = 0

        n_iter = len(loader)

        for data, label in tqdm(loader):
            pre = model(data.to(device))

            l = F.cross_entropy(target=label.to(device), input=pre)

            pre = pre.detach().cpu().numpy()
            ac = top_n(pre=pre, label=label.cpu().numpy(), n=top_n_acc)

            loss += l.item()
            acc += ac


        return loss / n_iter, acc / n_iter

    # def getRate(rateSample):
    #     np.cuda.Device(5).use()
    #     batchRate = []
    #     for i in range(len(rateSample)):
    #         _, rate = model(rateSample[i].unsqueeze(dim=0).to(device), return_rate=True)
    #         batchRate.append(rate)
    #
    #     return batchRate

    def getRate(rate_X, rate_Y):
        np.cuda.Device(Rdevice).use()
        _, rate = model(rate_X.to(device), rate_Y, return_rate=True)

        return rate # R(n_layer, 2)

    for i in range(epochSize):

        rate.update({i:getRate(rate_X, rate_Y)})

        loss, acc = cross_validation_epoch(
            model=model,
            loader=trainloader,
            optimizer=optimizer,
            top_n_acc=top_n_acc
        )

        loss_train.append(loss)
        acc_train.append(acc)

        loss, acc = estimate(
            model=model,
            loader=testloader,
            top_n_acc=top_n_acc
        )
        loss_test.append(loss)
        acc_test.append(acc)

        print("---Epoch {0}---\nLoss: --train{1} --test{2}\nAcc: --train{3} --test{4}".format(
            i+1, loss_train[-1], loss_test[-1], acc_train[-1], acc_test[-1]
        ))

    torch.save(
        obj={
            'model_state_dict':model.state_dict(),
            'acc_train': acc_train,
            'acc_test': acc_test,
            'loss_train': loss_train,
            'loss_test': loss_test,
            'rate': rate, # (epoch, n_rateSample, n_rate)
            'rate_sample': (rate_X,rate_Y)
        },
        f=logPATH + '/' + configName[:configName.find('.')] + '.pth'
    )

