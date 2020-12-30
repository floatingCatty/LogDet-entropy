import configparser
from trainer import train_NN
from utils import getConfig

if __name__ == '__main__':
    configName = getConfig()
    config = configparser.ConfigParser()
    config.read('./config/'+configName)

    model_parameters = config['model_parameters']
    others = config['others']

    train_NN(
        configName=configName,
        modelType=model_parameters['modelType'],
        lr=float(others['lr']),
        momentum=float(others['momentum']),
        n_class=int(model_parameters['n_class']),
        epochSize=int(others['epochSize']),
        n_train_sample=int(others['n_train_sample']),
        n_test_sample=int(others['n_test_sample']),
        batchSize=int(others['batchSize']),
        device=int(others['device']),
        dataset=others['dataset'],
        dataPATH=others['dataPATH'],
        logPATH=others['logPATH'],
        top_n_acc=int(others['top_n_acc'])
    )