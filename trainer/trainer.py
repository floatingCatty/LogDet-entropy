from model import resnet18, FNN, vgg13, AlexNet, LeNet, \
    CaLnet, NNencoder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import useData, top_n
import torch.nn.functional as F
import torchvision.models as models
import torch
from utils.LogDet import mi_loss
from tqdm import tqdm

def train(
        configName,
        model_parameters,
        modelType,
        rateSize,
        optim,
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
    rate_train = []
    rate_test = []
    channel_train = []
    channel_test = []

    if modelType == 'res18':
        model = resnet18(act=model_parameters['act'])
    elif modelType == 'fnn':
        model = FNN(act=model_parameters['act'])
    elif modelType == 'autoencoder':
        model = NNencoder(act=model_parameters['act'])
    elif modelType == 'vgg':
        model = vgg13(num_classes=n_class)
    elif modelType == 'densenet':
        model = models.densenet121()
    elif modelType == 'alexnet':
        model = AlexNet(num_classes=n_class, act=model_parameters['act'])
    elif modelType == 'lenet':
        model = LeNet(num_classes=n_class, in_channels=int(model_parameters['n_channel']))
    elif modelType == 'CaL':
        model = CaLnet(
            in_channel=int(model_parameters['in_channel']),
            num_classes=n_class,
            n_Layer=int(model_parameters['n_layer']),
            n_Channel=model_parameters['n_channel'],
            act=model_parameters['act']
        )
    else:
        raise ValueError

    loss_fn = F.cross_entropy

    model = model.train().to(device)

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        print(optim)
        raise ValueError

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    trainset, testset = useData(mode=dataset, PATH=dataPATH, transform=transform)

    train_X = torch.stack([trainset[i][0] for i in range(rateSize)]).to(device)
    train_Y = torch.tensor([trainset[i][1] for i in range(rateSize)])
    test_X = torch.stack([testset[i][0] for i in range(rateSize)]).to(device)
    test_Y = torch.tensor([testset[i][1] for i in range(rateSize)])

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
            pre = model(data.to(device), label, device=device)
            optimizer.zero_grad()


            l = loss_fn(target=label.to(device), input=pre)


            l.backward()
            optimizer.step()

            pre = pre.detach().cpu().numpy()
            ac = top_n(pre=pre, label=label.cpu().numpy(), n=top_n_acc)

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
            pre = model(data.to(device), label, device=device)

            l = loss_fn(target=label.to(device), input=pre)

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

        _, rate, Channel = model(x=rate_X, sample=rate_X, label=rate_Y, device=device, return_rate=True)

        return rate, Channel # R(n_layer, 2)

    for i in range(epochSize):
        rate, C = getRate(train_X, train_Y)
        rate_train.append(rate)
        channel_train.append(C)

        # print(rate_train[-1],rate_train[-1])
        rate, C = getRate(test_X, test_Y)
        rate_test.append(rate)
        channel_test.append(C)

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
        torch.cuda.empty_cache()
    # 'rate_train': torch.stack(rate_train), # (epoch, n_layer, k)
    torch.save(
        obj={
            'model_state_dict':model.state_dict(),
            'acc_train': acc_train,
            'acc_test': acc_test,
            'loss_train': loss_train,
            'loss_test': loss_test,
            'rate_test': torch.stack(rate_test),
            'rate_train': torch.stack(rate_train),
            'channel_train': torch.stack(channel_train),
            'channel_test': torch.stack(channel_test)
        },
        f=logPATH + '/network/'+ modelType + '/' + dataset + '/' + configName[:configName.find('.')] + '.pth'
    )

