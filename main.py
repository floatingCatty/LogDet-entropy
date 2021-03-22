import configparser
from trainer import train_IC
from utils import getConfig
import os
import sys

if __name__ == '__main__':
    configName = getConfig()
    config = configparser.ConfigParser()
    config.read('./config/'+configName, encoding='utf-8')

    model_parameters = config['model_parameters']
    others = config['others']
    train_IC(
        configName=configName,
        model_parameters=model_parameters,
        modelType=model_parameters['modelType'],
        rateSize=int(others['rateSize']),
        optim=others['optim'],
        lr=float(others['lr']),
        momentum=float(others['momentum']),
        n_class=int(model_parameters['n_class']),
        epochSize=int(others['epochSize']),
        n_train_sample=int(others['n_train_sample']),
        n_test_sample=int(others['n_test_sample']),
        device=str(others['device']),
        Rdevice=int(others['Rdevice']),
        dataset=others['dataset'],
        dataPATH=others['dataPATH'],
        logPATH=others['logPATH'],
        top_n_acc=int(others['top_n_acc']),
    )