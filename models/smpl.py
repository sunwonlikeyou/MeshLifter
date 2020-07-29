# code from Xiong Zhang
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import dataset.conf as conf
from models.util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose
import pdb

model_path = '/media/sunwon/Samsung_T5/MeshLifter/demo_meshlifter/models/neutral_smpl_with_cocoplus_reg.pkl'
#model_path = '/home/sunwon/Dropbox/Dropbox/projects/MeshLifter/src/models/neutral_smpl_with_cocoplus_reg.pkl'
class SMPL(nn.Module):
    def __init__(self, joint_type = 'cocoplus', obj_saveable = False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        # load SMPL parameters
        self.joint_type = joint_type
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
 
        # maintain face information?
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None
        # mean template vertices (6890 x 3)
        np_v_template = np.array(model['v_template'], dtype = np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        # shape blend shape basis (10 x 20670)
        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        # regressor for joint locations given shape (6890 x 24)
        np_J_regressor = np.array(model['J_regressor'].T.todense(), dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        # pose blend shape basis (207 x 20670)
        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        # indices of parents for each joints
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        # return 19 or 14 keypoints (6890 x 19 or 14)
        np_joint_regressor = np.array(model['cocoplus_regressor'].T.todense(), dtype = np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        # LBS weights (6890 x 24)
        np_weights = np.array(model['weights'], dtype = np.float)

        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]

        batch_size = conf.max_batch_size
        np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, theta, trans, get_skin = False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        # 1. add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890 x 3
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template

        # 2. infer shape-dependent joint locations
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)

        # 3. add pose blend shapes
        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        # 4. get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = True)

        # 5. do skinning
        weight = self.weight[:num_batch]
        # W is N x 6890 x 24
        W = weight.view(num_batch, -1, 24)
        # (N x 6890 x 24) x (N x 24 x 16)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        # 6. translation
        verts = verts + torch.reshape(trans, (num_batch, 1, 3))
        #pdb.set_trace()

        # 7. get cocoplus or lsp joints
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

if __name__ == '__main__':
    device = torch.device('cuda', 0)
    smpl = SMPL(obj_saveable = True).to(device)
    pose= np.array([
            1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
            -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
            2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
            7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
            -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
            6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
            -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
            4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
            2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
            -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
            -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
            3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)

    beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
            0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])

    vbeta = torch.tensor(np.array([beta])).float().to(device)
    vpose = torch.tensor(np.array([pose])).float().to(device)

    verts, j, r = smpl(vbeta, vpose, get_skin = True)
    print(j)
    pdb.set_trace()

    smpl.save_obj(verts[0].cpu().numpy(), './mesh.obj')

    rpose = reflect_pose(pose)
    vpose = torch.tensor(np.array([rpose])).float().to(device)
    
    verts, j, r = smpl(vbeta, vpose, get_skin = True)
    smpl.save_obj(verts[0].cpu().numpy(), './rmesh.obj')

