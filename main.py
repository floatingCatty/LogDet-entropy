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

        e=model_parameters['e'],
        nameta=model_parameters['nameta'],
        n_class=model_parameters['n_class'],
        n_channel=model_parameters['n_channel'],
        in_channel=model_parameters['in_channel'],
        kernel_size=model_parameters['kernel_size'],
        L=model_parameters['L'],
        lr=model_parameters['lr'],

        dataPATH=others['dataPATH'],
        logPATH=others['logPATH'],
        device=others['device'],
        dataset=others['dataset'],
        n_train_sample=others['n_train_sample'],
        n_test_sample=others['n_test_sample'],
        update_batchsize=others['update_batchsize'],
        approximation=others['approximation'],
        top_n_acc=others['top_n_acc']
    )