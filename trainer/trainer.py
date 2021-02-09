from model import ReduNet_2D
import cupy as np
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader, random_split
from utils import useData, top_n
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
        modelType,

        lr,
        momentum,
        n_class,
        epochSize,
        n_train_sample,
        n_test_sample,
        batchSize,
        device,
        dataset,
        dataPATH,
        logPATH,
        top_n_acc
):
    '''
    modelType: model supported in torchvision
    '''

    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []

    if modelType == 'res18':
        model = models.resnet18(pretrained=False, progress=True, num_classes=n_class)
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

    if n_class == 2:
        trainset_indices = ((trainset.targets == 0) + (trainset.targets == 1)).nonzero().view(-1)

        trainloader = DataLoader(
            dataset=trainset,
            batch_size=n_train_sample,
            shuffle=False,
            sampler=SubsetRandomSampler(trainset_indices)
        )

        testset_indices = ((testset.targets == 0) + (testset.targets == 1)).nonzero().view(-1)

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

    for train_data, train_label in trainloader:
        break
    for test_data, test_label in testloader:
        break

    train_data = train_data.to(device)
    train_label = train_label.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)

    N, C, H, W = train_data.size()
    n, _, _, _ = test_data.size()
    if C == 1:
        train_data = train_data.expand(N, 3, H, W)
        test_data = test_data.expand(n, 3, H, W)



    def cross_validation_epoch(
            model,
            data,
            label,
            batchSize,
            optimizer,
            top_n_acc
    ):
        loss = 0
        acc = 0

        n_iter = int(len(label) / batchSize) + 1

        for i in range(n_iter):
            pre = model(data[i:min((i+1)*batchSize, len(data))])
            optimizer.zero_grad()

            l = F.cross_entropy(target=label[i:min((i + 1) * batchSize, len(data))], input=pre)

            pre = pre.detach().cpu().numpy()
            ac = top_n(pre=pre, label=label[i:min((i + 1) * batchSize, len(data))].cpu().numpy(), n=top_n_acc)

            l.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            loss += l.item()
            acc += ac

        return loss / n_iter, acc / n_iter

    def estimate(
            model,
            data,
            label,
            top_n_acc
    ):
        loss = 0
        acc = 0

        n_iter = int(len(label) / batchSize) + 1

        for i in range(n_iter):
            pre = model(data[i:min((i + 1) * batchSize, len(data))])

            l = F.cross_entropy(target=label[i:min((i + 1) * batchSize, len(data))], input=pre)

            pre = pre.detach().cpu().numpy()
            ac = top_n(pre=pre, label=label[i:min((i + 1) * batchSize, len(data))].cpu().numpy(), n=top_n_acc)

            loss += l.item()
            acc += ac


        return loss / n_iter, acc / n_iter

    for i in tqdm(range(epochSize)):
        loss, acc = cross_validation_epoch(
            model=model,
            data=train_data,
            label=train_label,
            batchSize=batchSize,
            optimizer=optimizer,
            top_n_acc=top_n_acc
        )

        loss_train.append(loss)
        acc_train.append(acc)

        loss, acc = estimate(
            model=model,
            data=test_data,
            label=test_label,
            top_n_acc=top_n_acc
        )
        loss_test.append(loss)
        acc_test.append(acc)

        print(loss_train[-1], loss_test[-1], acc_train[-1], acc_test[-1])

    torch.save(
        obj={
            'model_state_dict':model.state_dict(),
            'acc_train': acc_train,
            'acc_test': acc_test,
            'loss_train': loss_train,
            'loss_test': loss_test
        },
        f=logPATH + '/' + configName[:configName.find('.')] + '.pth'
    )

