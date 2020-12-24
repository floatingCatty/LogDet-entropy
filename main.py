import configparser
from trainer import train_2D
from utils import getConfig

if __name__ == '__main__':
    configName = getConfig()
    config = configparser.ConfigParser()
    config.read('./config/'+configName)

    model_parameters = config['model_parameters']
    others = config['others']
    train_2D(
        configName=configName,

        e=float(model_parameters['e']),
        nameta=float(model_parameters['nameta']),
        n_class=int(model_parameters['n_class']),
        n_channel=int(model_parameters['n_channel']),
        in_channel=int(model_parameters['in_channel']),
        kernel_size=(int(model_parameters['kernel_size_h']),int(model_parameters['kernel_size_w'])),
        L=int(model_parameters['L']),
        lr=float(model_parameters['lr']),

        dataPATH=others['dataPATH'],
        logPATH=others['logPATH'],
        device=int(others['device']),
        dataset=others['dataset'],
        n_train_sample=int(others['n_train_sample']),
        n_test_sample=int(others['n_test_sample']),
        update_batchsize=int(others['update_batchsize']),
        label_proportion=float(others['label_proportion']),
        top_n_acc=int(others['top_n_acc'])
    )