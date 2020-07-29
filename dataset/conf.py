# data directory
data_dir = '/media/sunwon/Samsung_T5/MeshLifter/demo_meshlifter/dataset'
# image directory
h36m_img_dir = '/media/sunwon/Samsung_T5/H36M/images'
#h36m_img_dir = '/home/sun/container_sun/H36M/images'

mpii_img_dir = '/home/sun/ssd_sun/MPII/images'
#mpii_img_dir = '/home/sun/container_sun/MPII/images'
inf_img_dir = './data/inf'

# experiment directory
exp_dir = '/media/sunwon/Samsung_T5/MeshLifter/demo_meshlifter/dataset/'
hand_exp = '/home/sunwon/Dropbox/Dropbox/projects//hand_exp'


# number of threads
num_threads = 20
hand_threads = 4

# number of joints
num_joints_in = 17
num_joints_out = 14

hand_joints = 21
# root index (hip)
root = 0

# input/output resolutions
res_in = 256
res_out = 64

# standard deviation of gaussian function for heatmap generation
std = 1.0

# parameters for data augmentation
scale = 0.25
rotate = 30
flip_index = [[3, 6], [2, 5], [1, 4],
              [16, 13], [15, 12], [14, 11]]

# joint index mapping from MPII to H36M
inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]

# number of actions for human3.6m dataset
num_actions = 15

# max batch size
max_batch_size = 1024

