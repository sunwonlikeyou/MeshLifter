import os
import torch
from utils.utils import AverageMeter
from utils.eval import compute_error3d, compute_error2d
from utils.eval import compute_error3d_pa_simil
from progress.bar import Bar
import pdb

def _weighted_mse_loss(prediction, target, weight = 1.0):
    return torch.sum(weight*(prediction-target)**2)/prediction.shape[0]

def _weighted_l1_loss(prediction, target):
    return torch.sum(torch.abs(prediction-target))/(prediction.shape[0]*prediction.shape[1]*prediction.shape[2])

def _adv_l2_loss(prediction):
    return torch.sum((prediction-1.0)**2.0)/(prediction.shape[0]*prediction.shape[1])

def _adv_l1_loss(prediction):
    return torch.sum(torch.abs(prediction-1.0))/(prediction.shape[0]*prediction.shape[1])

def _disc_l2_loss(real, fake):
    return torch.sum((real-1.0)**2.0)/(real.shape[0]*real.shape[1]) + torch.sum(fake**2.0)/(fake.shape[0]*fake.shape[1])

def _disc_l2_loss_2(real, fake1, fake2):
    return torch.sum((real-1.0)**2.0)/(real.shape[0]*real.shape[1]) \
        + torch.sum(fake1**2.0)/(fake1.shape[0]*fake1.shape[1]) \
        + torch.sum(fake2**2.0)/(fake2.shape[0]*fake2.shape[1])

def _normalize_pose(x):
    nb = x.shape[0]
    mean2d = torch.mean(x, 1, keepdim=True)
    dist = torch.sqrt(torch.sum((x-mean2d)**2.0, 2, keepdim=True))
    std2d = torch.std(dist, 1, keepdim=True)
    x = (x - mean2d) / std2d
    return x

def _project_orthographic(j3d):
    j2d = j3d[:,:,:2].clone()
    return j2d

def test(split, epoch, loader_joint, generator):

    error2d, error3d = AverageMeter(), AverageMeter()
    error3d_pa = AverageMeter()

    num_iters = len(loader_joint)
    bar = Bar('==>', max=num_iters)

    # for each mini-batch,
    for i, (data) in enumerate(loader_joint):
        pose2d = data['pose2d'].float().to("cuda")
        bbox = data['bbox'].float().to("cuda")
        pose3d = data['pose3d'].to("cuda")
        
        
        rot = data['rot'].to("cuda").detach()
        rot_inv = data['rot_inv'].to("cuda").detach()
        nb = pose2d.shape[0]

        # get normallized feature
        pose2d_in = _normalize_pose(pose2d)

        # forward propagation
        generator_output = generator(pose2d_in)
        (thetas_out, verts_out, pose3d_out) = generator_output
        pose_1st = thetas_out[:,13:].clone()
        shape_1st = thetas_out[:,0:10].clone()
        shape_1st = shape_1st.unsqueeze(dim=2) #for weighted l1 loss



        error3d.update(compute_error3d(pose3d_out.detach(), pose3d))

        if split == 'test':
            error3d_pa.update(compute_error3d_pa_simil(pose3d_out.detach(), pose3d))

        Bar.suffix='Reconstruction Error {error3d_pa.avg:.2f}'.format(error3d_pa=error3d_pa)
        bar.next()

    print("3D PA error for test set (PA-MPJPE)= %.6f\n" % (error3d_pa.avg))


    out = {'error3d': error3d.avg, 'error2d': error2d.avg}
    return out

