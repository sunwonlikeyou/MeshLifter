import torch
import torch.nn as nn
from models.smpl import SMPL
import dataset.conf
import pdb

# dimension of theta: rotation and shape
dim_theta = 24*3 + 10

class ResNetModule(nn.Module):
    def __init__(self, num_features):
        super(ResNetModule, self).__init__()

        modules = []
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))

        # set weights
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.submod = nn.Sequential(*modules)

    def forward(self, x):
        return self.submod(x) + x

class ResNet(nn.Module):
    def __init__(self, num_features):
        super(ResNet, self).__init__()

        self.num_in = 2*14
        self.num_out = dim_theta
        self.num_features = num_features

        self.linear1 = nn.Linear(self.num_in, num_features)
        self.mod1 = ResNetModule(num_features)
        self.mod2 = ResNetModule(num_features)
        self.linear2 = nn.Linear(num_features, self.num_out)
        
        # set weights
        nn.init.normal_(self.linear1.weight, mean=0, std=0.001)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.001)
        nn.init.constant_(self.linear2.bias, 0)
       

        #self.smpl = smpl = SMPL(joint_type='lsp')
        self.smpl = smpl = SMPL(joint_type='lsp', obj_saveable=True)

    def forward(self, x):
        nb = x.shape[0]
        x = x.reshape(nb, -1)
        # feed normalized features to network
        x = self.linear1(x)
        x = self.mod1(x)
        x = self.mod2(x)
        x = self.linear2(x)

        thetas = x[:, 0:82].clone()
        betas = x[:, 0:10].clone()
        poses = x[:, 10:82].view(-1, 24, 3).clone()

        verts, j3d, Rs = self.smpl(beta=betas, theta=poses, trans=torch.zeros(nb,3).to(x.device), get_skin=True)
        verts = verts * 1000.
        j3d = j3d * 1000.

        j3d_root = (j3d[:,2,:]+j3d[:,3,:])*.5
        j3d = j3d - j3d_root.reshape(nb,1,3)
        verts = verts - j3d_root.reshape(nb,1,3)
        #pdb.set_trace()
        

        return (thetas, verts, j3d)

