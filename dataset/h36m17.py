import torch
import torch.utils.data as data
from h5py import File
import dataset.conf as conf
#import conf
import math
import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2
from utils.utils import rnd, flip, shuffle_lr
#from utils.utils import rnd
#from utils import rnd,flip,shuffle_lr
from mpl_toolkits.mplot3d import axes3d, Axes3D

subject_list = [[[1, 5, 6, 7], [8]], [[1, 5, 6, 7, 8], [9, 11]], [[1, 5, 6, 7, 8, 9], [11]]]
flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
flip_index_2 = [5,4,3,2,1,0,11,10,9,8,7,6,12,13]
# index mapping from h36m (17 joints) to lsp (14 joints)
inds = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

img_dir = '/home/sun/container_sun/H36M/images'
bones = np.array([[0,1,2,3], [3,4,5], [6,7,8,9], [9,10,11], [12,13]]) #bones와 이 숫자들의 의미
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
def draw_skeleton(img, joints):
    bone = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9, ], [9, 10], [10, 11], [12, 13]]
    thickness = 3
    radius = 5
    for i in range(joints.shape[0]):
        cv2.circle(img, (int(joints[i, 0] + 0.5), int(joints[i, 1] + 0.5)), 5, (0, 255, 255), -1)

    for i in range(len(bone)):
        a, b = bone[i][0], bone[i][1]
        cv2.line(img, (int(joints[a, 0] + 0.5), int(joints[a, 1] + 0.5)),
                 (int(joints[b, 0] + 0.5), int(joints[b, 1] + 0.5)), (0, 255, 0), thickness)

    return img

import pdb
def plot_3d_bone(pose3d):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.plot(pose3d[bones[0],0], pose3d[bones[0],2], -pose3d[bones[0],1], lw=5, c='r')#red
    ax.plot(pose3d[bones[1],0], pose3d[bones[1],2], -pose3d[bones[1],1], lw=5, c='b')# blue
    ax.plot(pose3d[bones[2],0], pose3d[bones[2],2], -pose3d[bones[2],1], lw=5, c='g')# green
    ax.plot(pose3d[bones[3],0], pose3d[bones[3],2], -pose3d[bones[3],1], lw=5, c='m')# violet
    ax.plot(pose3d[bones[4],0], pose3d[bones[4],2], -pose3d[bones[4],1], lw=5, c='c')# mint

    #ax.set_aspect('equal')
    #axisEqual3D(ax)
    plt.show()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=np.float32)

def draw_joint(img, joint):
    thickness = 4
    radius = 5
    #pdb.set_trace()
    img = cv2.circle(img, (int(joint[0]+0.5),int(joint[1]+0.5)),thickness,(255,0,0,),-1)
    return img

class H36M17(data.Dataset):
    def __init__(self, protocol, split, dense=False, scale=False, noise=0, std_train=0, std_test=0, noise_path=None):
        print('==> Initializing H36M %s data' % (split))
        annot = {}
        tags = ['idx', 'pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'cam_R', 'cam_T',
                'subject', 'action', 'subaction', 'camera']
        if split == 'train':
            #f = File('%s/h36m/protocol2/h36m17_protocol%d.h5' % (conf.data_dir,protocol), 'r')
            f = File('%s/data/h36m17_new.h5' % (conf.data_dir), 'r')
        elif split == 'test' or split == 'val': 
            f = File('%s/data/h36m17_protocol2_4.h5' % (conf.data_dir), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()
        if dense == False:
            idxs = np.mod(annot['idx'], 50) == 1
            idxs = np.arange(annot['idx'].shape[0])[idxs]
            for tag in tags:
                annot[tag] = annot[tag][idxs]
    
       
        idxs = np.full(annot['idx'].shape[0], False)
        subject = subject_list[protocol-1][1-int(split=='train' or split=='test_train')]
        for i in range(len(subject)):
            idxs = idxs + (annot['subject']==subject[i])
        idxs = np.arange(annot['idx'].shape[0])[idxs]
        for tag in tags:
            annot[tag] = annot[tag][idxs]

        self.protocol = protocol
        self.split = split
        self.dense = dense
        self.scale = scale
        self.noise = noise
        self.std_train = std_train
        self.std_test = std_test
        self.noise_path = noise_path
        self.annot = annot
        self.num_samples = len(self.annot['idx'])

        # image size
        self.width = 256
        self.height = 256

        # load error statistics
        self.load_error_stat()

        print('Load %d H36M %s samples' % (self.num_samples, self.split))

    def get_part_info(self, index):
        pose2d = self.annot['pose2d'][index].copy()
        #print(len(self.annot['pose2d']), self.split)
        bbox = self.annot['bbox'][index].copy()
        pose3d = self.annot['pose3d'][index].copy()
        cam_f = self.annot['cam_f'][index].copy()
        cam_c = self.annot['cam_c'][index].copy()
        cam_R = self.annot['cam_R'][index].copy()
        cam_T = self.annot['cam_T'][index].copy()
        return pose2d, bbox, pose3d, cam_f, cam_c, cam_R, cam_T
    
    def load_image(self, index):
        dirname = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(self.annot['subject'][index], \
            self.annot['action'][index], self.annot['subaction'][index], self.annot['camera'][index])
        imgname = '{}/{}/{}_{:06d}.jpg'.format(conf.h36m_img_dir, dirname, dirname, self.annot['idx'][index])
        img = cv2.imread(imgname)
        return img

    def load_error_stat(self):
        # load error stat
        if self.split == 'train':
            if self.noise == 0: # do not use noise
                print("success loading %s GT data"%(self.split))
                pass
            elif self.noise == 1: # use specified gaussian noise
                pass
            elif self.noise == 2: # use estimated 2d pose
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                #print(filename)
                result = torch.load(filename)
                self.annot['pose2d'] = result['pred'].cpu().numpy()
                print("success loading %s Estimated data"%(self.split))

            elif self.noise == 3: # use estimated single gaussian noise
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                result = torch.load(filename)
                mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.mean = mean[0]
                std = result['std'].numpy() / float(conf.res_in - 1)
                self.std = std[0]
            elif self.noise == 4: # use estimated mixture noise
                filename = '%s/%s' % (conf.exp_dir, self.noise_path)
                result = torch.load(filename)
                self.mean = result['mean'].numpy() / float(conf.res_in - 1)
                self.std = result['std'].numpy() / float(conf.res_in - 1)
                self.weight = result['weight'].numpy()
            else:
                raise ValueError('unsupported noise mode %d' % self.noise)
        elif self.split == 'test':
            filename = '%s/data/%s' % (conf.exp_dir, self.noise_path)
            print("success loading %s GT data"%(self.split))
            #print(filename)
            result = torch.load(filename)
            self.annot['pose2d'] = result['pred2d'].cpu().numpy()

    def __getitem__(self, index):
        img  = self.load_image(index)
        # get 2d/3d pose, bounding box, camera information
        pose2d, bbox, pose3d, cam_f, cam_c, cam_R, cam_T = self.get_part_info(index)
        # original 2d pose
        meta2d = pose2d.copy()
        cam_f = cam_f.astype(np.float32)
        cam_c = cam_c.astype(np.float32)
        cam_R = cam_R.reshape((3,3)).astype(np.float32)
        cam_T = cam_T.astype(np.float32)

        # induce scale variation?
        if self.scale == True:
            s = 2 ** rnd(0.25)
            pose2d = pose2d * s
            bbox = bbox * s
            cam_f = cam_f * s
            cam_c = cam_c * s
            width = self.width * s
        else:
            width = self.width


        # lsp
        if (self.noise == 0 and self.split == 'train') or self.split == 'val':
            pose2d = pose2d[inds].copy()
         

        # set 3d pose
        pose3d = pose3d[inds].copy()
        root3d = (pose3d[2,:]+pose3d[3,:])*.5
        pose3d = pose3d - root3d.reshape(1,3)

        # rotation matrix
        # 1. local -> global: cam_R
        # 2. random rotation in z-axis
        # 3. global -> local: np.transpose(cam_R)
        theta0 = -60.
        theta1 = 60.
        theta = np.random.rand()*(theta1-theta0) + theta0
        theta = theta * np.pi / 180.
        axis = np.array([0, 0, 1], dtype=np.float32)
        rot = rotation_matrix(axis, theta)
        rot = np.matmul(np.transpose(cam_R), np.matmul(rot, cam_R))
        rot_inv = np.transpose(rot)

        # matrix multiplication on right side
        rot = np.transpose(rot)
        rot_inv = np.transpose(rot_inv)
        bbox = bbox.astype(np.float32)
        pose2d = pose2d.astype(np.float32)
        # set data
        data = {'pose2d': pose2d, 'bbox': bbox , 'img':img,
                    'pose3d': pose3d,
                    'rot': rot, 'rot_inv': rot_inv}

        return data

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import torch
    import cv2
    import numpy as np

    noise_train_h36m = 'test_train_H36M.pth'
    noise_h36m = 'result_H36M.pth'
    dataset = H36M17(2, 'test', True, False, 0, 0,0,noise_h36m)
    l = len(dataset)



    for _ in range(l):
        data = dataset.__getitem__(_)
        pose2d = data['pose2d']
        bbox = data['bbox']
        pose3d = data['pose3d']
        rot = data['rot']
        rot_inv = data['rot_inv']
        img = data['img']

        print(pose3d)
        print(pose2d)

        img = draw_skeleton(img, pose2d)
        cv2.imshow('original', img)
        # cv2.imshow('mesh', rn.r)
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
            pdb.set_trace()
            break
        #plot_3d_bone(pose3d)




