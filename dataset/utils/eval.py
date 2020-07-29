import torch
import dataset.conf
import pdb

def get_pred(hmap):
    """get predicted 2d pose from heat map"""
    num_batch = hmap.shape[0]
    num_joint = hmap.shape[1]
    h = hmap.shape[2]
    w = hmap.shape[3]
    hmap = hmap.reshape(num_batch, num_joint, h*w)
    idx = torch.argmax(hmap, dim=2)
    pred = torch.zeros(num_batch, num_joint, 2).to('cuda')
    for i in range(num_batch):
        for j in range(num_joint):
            pred[i, j, 0], pred[i, j, 1] = idx[i, j] % w, idx[i, j] / w
    return pred

def compute_error(output, target, weight):
    """compute 2d pose estimation error in pixels"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    res_in = conf.res_in
    res_out = output.shape[2]
    res_ratio = res_in / res_out
    pred = get_pred(output)
    pred = pred * res_ratio + res_ratio/2
    val = torch.sqrt(torch.mul(((pred - target) ** 2).sum(2), weight))
    error = val.sum()/weight.sum().item()
    return error

def compute_error_direct(output, target, weight,split):
    """compute 2d pose estimation error in pixels"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    val = torch.sqrt(torch.mul(((output - target) ** 2).sum(2), weight))
    error = val.sum()/weight.sum().item()
    return error


def compute_error2d(output, target):
    """compute 2D pose estimation error in pixels"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    val = torch.sqrt(((output - target) ** 2).sum(2))
    error = val.sum()/(num_batch*num_joint)
    return error

def compute_error3d(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    val = torch.sqrt(((output - target) ** 2).sum(2))
    error = val.sum()/(num_batch*num_joint)
    return error

def compute_error3d_pa_simil(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]

    # translation
    output_mean = output.mean(dim=1, keepdim=True)
    output = output - output_mean
    target_mean = target.mean(dim=1, keepdim=True)
    target = target - target_mean

    # scaling
    output_scale = torch.sqrt((output**2.0).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)/num_joint)
    target_scale = torch.sqrt((target**2.0).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)/num_joint)
    scale_ratio = target_scale / output_scale
    output = output * scale_ratio

    # rotation (solve orthogonal Procrustes problem)
    output_pa = output.clone()
    for i in range(num_batch):
        A = output[i].t()
        B = target[i].t()
        M = torch.mm(B, A.t())
        U, S, V = torch.svd(M)
        R = torch.mm(U, V.t())
        A = torch.mm(R, A)
        output_pa[i] = A.t()

    # compute MPJPE error
    val = torch.sqrt(((output_pa - target) ** 2).sum(2))
    error = val.sum()/(num_batch*num_joint)

    return error

def compute_error3d_pa_rigid(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]

    # translation
    output_mean = output.mean(dim=1, keepdim=True)
    output = output - output_mean
    target_mean = target.mean(dim=1, keepdim=True)
    target = target - target_mean

    # rotation (solve orthogonal Procrustes problem)
    output_pa = output.clone()
    for i in range(num_batch):
        A = output[i].t()
        B = target[i].t()
        M = torch.mm(B, A.t())
        U, S, V = torch.svd(M)
        R = torch.mm(U, V.t())
        A = torch.mm(R, A)
        output_pa[i] = A.t()

    # compute MPJPE error
    val = torch.sqrt(((output_pa - target) ** 2).sum(2))
    error = val.sum()/(num_batch*num_joint)

    return error

def compute_error3d_x(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]+1
    val = torch.abs(output[:,:,0] - target[:,:,0])
    error = val.sum()/(num_batch*num_joint)
    return error

def compute_error3d_y(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]+1
    val = torch.abs(output[:,:,1] - target[:,:,1])
    error = val.sum()/(num_batch*num_joint)
    return error

def compute_error3d_z(output, target):
    """compute 3D pose estimation error in millimeters"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]+1
    val = torch.abs(output[:,:,2] - target[:,:,2])
    error = val.sum()/(num_batch*num_joint)
    return error

