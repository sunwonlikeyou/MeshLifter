import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pdb
import dataset.conf as conf
from dataset.h36m17 import H36M17
from models.resnet import ResNet
from test_util import test
from utils.logger import Logger
import os


def main():
    # for repeatable experiments
    torch.backends.cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    

    # gpus
    gpus = [0]

    noise_h36m = 'result_H36M.pth'
    # --------------------------------------------------------------------
    # test loader for final prediction
    loader_test = torch.utils.data.DataLoader(
            dataset=H36M17(2, 'test', False, False, 2, 0.0, 0.0, noise_h36m),
            batch_size=512* len(gpus),
            shuffle=False,
            num_workers=conf.num_threads
        )


    # build models
    #device = torch.device("cuda:1")
    generator = ResNet(3000).cuda()
    generator = nn.DataParallel(generator, device_ids=gpus)
    generator.eval()

    save_dir = '/media/sunwon/Samsung_T5/MeshLifter/demo_meshlifter' # directory of final model.pth

    file_name = os.path.join(save_dir, 'final_model.pth')
    if os.path.exists(file_name):
        state = torch.load(file_name)
        generator.load_state_dict(state['generator'])
        print('success model loading')
    else:
        print('Doesnt exist!')



    # generate final prediction
    with torch.no_grad():
        test('test',1, loader_test, generator)


if __name__ == '__main__':
    main()


