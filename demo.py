import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
import pyrender
import trimesh
import numpy as np
import dataset.conf
from models.resnet import ResNet
from dataset.h36m17 import H36M17
import pdb
import cv2
#from opendr.renderer import ColoredRenderer
#from opendr.lighting import LambertianPointLight
#from opendr.camera import ProjectPoints
#from vis_template import display_hand
from mpl_toolkits.mplot3d import axes3d, Axes3D
from utils.img import transform, crop, draw_gaussian
from matplotlib import pyplot as plt
#from mpii import MPII
#from inf import MPIINF

bones = np.array([[0,1,2,3], [3,4,5], [6,7,8,9], [9,10,11], [12,13]]) #bones와 이 숫자들의 의미

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


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

def _overlap_def(verts, faces):
    fx = 285.
    fy = 285.
    cx = 160.
    cy = 160.

    width = 480
    height = 480
    # Set renderer
    rn = ColoredRenderer()
    rn.camera = ProjectPoints(v=verts, rt=np.ones(3), t=np.array([0, 0, 500]), f=np.array([fx, fy]),
                              c=np.array([cx, cy]), k=np.zeros(5))
    rn.frustum = {'near': 100., 'far': 800., 'width': width, 'height': height}
    rn.set(v=verts, f=faces, bgcolor=np.ones(3))

    # Construct point light source
    rn.vc = LambertianPointLight(
        f=faces,
        v=rn.v,
        num_verts=verts.shape[0],
        light_pos=np.array([-1000, -1000, -2000]),
        vc=np.ones_like(verts) * .9,
        light_color=np.array([1., 1., 1.]))
    return rn

def _normalize_pose(x):
    nb = x.shape[0]
    mean2d = torch.mean(x,1, keepdim = True)
    dist = torch.sqrt(torch.sum((x-mean2d)**2.0,2, keepdim = True))
    std2d = torch.std(dist, 1, keepdim = True)
    x = (x-mean2d)/std2d
    return x

def normalize_np_pose(x):
    nb = x.shape[0]
    mean2d = np.mean(x,0, keepdims = True)
    dist = np.sqrt(np.sum((x-mean2d)**2.0,1, keepdims = True))
    std2d = np.std(dist, 0, keepdims = True)
    x = (x-mean2d)/std2d
    return x, mean2d, std2d

def draw_joint(img, joint):
    thickness = 4
    radius = 5
    #pdb.set_trace()
    img = cv2.circle(img, (int(joint[0]+0.5),int(joint[1]+0.5)),thickness,(255,0,0,),-1)
    return img

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

noise_h36_val = 'result_H36M.pth'

def main(db):
    if db == 'h36m':
        loader_val = torch.utils.data.DataLoader(
                dataset = H36M17(2, 'test', False, False, 2, 0, 0, noise_h36_val),
                batch_size =1,
                shuffle =False,
                num_workers =0
                )
    model = ResNet(3000).cuda()
    model = nn.DataParallel(model, device_ids=[0])

    #last full model
    
    save_dir = '/media/sunwon/Samsung_T5/MeshLifter/demo_meshlifter/'

    filename = '%s/final_model.pth'%(save_dir)
    state = torch.load(filename)
    model.load_state_dict(state['generator'], strict = False)
    model.eval()

    for i,data in enumerate(loader_val):
        if np.mod(i, 1) ==0:
            pose2d = data['pose2d'].float().to("cuda")
            print(pose2d)
            bbox = data['bbox'].float().to("cuda")
            pose3d = data['pose3d'].float().to("cuda")
            rot = data['rot'].to("cuda").detach()
            rot_inv = data['rot_inv'].to("cuda").detach()
            img = data['img'].detach().cpu().numpy().squeeze()
            #img = cv2.resize(img, (256,256))
            meta2d = pose2d[0].clone()
            
            faces = model.module.smpl.faces
            pose2d_in  = _normalize_pose(pose2d)
            rot = rot[0].detach().cpu().numpy()

            generator_output = model(pose2d_in)
            (thetas_out, verts_out, pose3d_out) = generator_output

            pose3d = pose3d.detach().cpu().numpy().squeeze()
            pose3d_out = pose3d_out.detach().cpu().numpy().squeeze()
            
            verts = verts_out[0].detach().cpu().numpy()
            pose2d_rot = pose3d_out[:,:2]
            
            pose2d = pose2d[0].detach().cpu().numpy()

            pose2d , mean, std = normalize_np_pose(pose2d)
            pose2d_rot,a,b = normalize_np_pose(pose2d_rot)
            pose2d_rot = pose2d_rot*std + mean
            vertex_color = np.ones([verts.shape[0],4])*[0.8,0.8,0.8,1.0]
            tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors = vertex_color)

            pts_color = np.ones([pose3d_out.shape[0],4])
            pts3d= pyrender.Mesh.from_points(pose3d_out, colors =pts_color)


            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene = pyrender.Scene()
            scene.add(mesh)
            
            pose2d = pose2d[0] # batch 14 2  -> 14 2
            len = pose2d.shape[0]
            
            ori_img = img.copy()

            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_2d = draw_skeleton(img, meta2d)
            print(meta2d)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('original', img_2d)

            # cv2.imshow('mesh', rn.r)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                pdb.set_trace()
                break
            elif k == ord('s'):
                cv2.destroyAllWindows()

            pyrender.Viewer(scene, use_raymond_lighting = True)

            verts = np.expand_dims(verts,axis=0)
            joints = np.expand_dims(pose3d_out, axis=0)



main('h36m')


