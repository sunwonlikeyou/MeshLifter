import argparse
import os
import conf
import pdb
import time

now = time.gmtime(time.time())

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def init(self):
        # miscellaneous
        self.parser.add_argument('-dataset_test', default='h36m', help='Test dataset')
        self.parser.add_argument('-dataset_train', default='h36m', help='Training dataset')
        self.parser.add_argument('-protocol', type=int, default=1, help='Experiment protocol for H36M: 0 | 1 | 2')
        self.parser.add_argument('-self', default=False, action='store_true', help='Self-supervised learning?')
        self.parser.add_argument('-loop', default=False, action='store_true', help='Loop?')
        self.parser.add_argument('-noise', type=int, default=0, help='Noise mode')
        self.parser.add_argument('-noise_path', default=None, help='Path to noise info')
        self.parser.add_argument('-std_train', type=float, default=0.0, help='Std of Gaussian noise for robust training')
        self.parser.add_argument('-std_test', type=float, default=0.0, help='Std of Gaussian noise for testing')
        self.parser.add_argument('-scale', default=False, action='store_true', help='Induce random scaling for data augmentation?')
        self.parser.add_argument('-idea', default = 'Nothing', type = str, help = 'what is the idea')
        self.parser.add_argument('-sja', default = 0.0 , type =float , help= 'Use sja module')

        # network structure
        self.parser.add_argument('-network', default='resnet', help='Network to use')
        self.parser.add_argument('-num_layers', type=int, default=2, help='Number of hidden layers')
        self.parser.add_argument('-num_features', type=int, default=3000, help='Number of features')
        
        # loss
        self.parser.add_argument('-weight_3d', type=float, default=0e1, help='Weight for 3D supervision')
        self.parser.add_argument('-weight_2d', type=float, default=1e1, help='Weight for discriminator loss of generator')
        self.parser.add_argument('-weight_adv_mesh', type=float, default=1e3, help='Weight for discriminator loss of generator')
        self.parser.add_argument('-weight_disc_mesh', type=float, default=1e3, help='Weight for discriminator loss or discriminator')
        self.parser.add_argument('-weight_adv_2d', type=float, default=1e3, help='Weight for discriminator loss of generator')
        self.parser.add_argument('-weight_disc_2d', type=float, default=1e3, help='Weight for discriminator loss or discriminator')
        self.parser.add_argument('-weight_reg',type = float, default = 0e0, help= 'Weight for Regulazriation / template mesh')
        self.parser.add_argument('-weight_pose_con',type = float, default = 0e0, help= 'Weight for pose consistency')
        self.parser.add_argument('-weight_loop',type = float, default = 1e1, help= 'Weight for loop consistency')



        # optimization
        self.parser.add_argument('-opt_method', default='adam', help='Optimization method')
        self.parser.add_argument('-lr_g', type=float, default=1e-4, help='Learning rate for generator')
        self.parser.add_argument('-lr_d', type=float, default=1e-4, help='Learning rate for discriminator')
        self.parser.add_argument('-alpha', type=float, default=0.99, help='Smoothing constant')
        self.parser.add_argument('-epsilon', type=float, default=1e-8, help='For numerical stability')
        self.parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay')
        self.parser.add_argument('-lr_decay', type=float, default=0, help='Learning rate decay')
        self.parser.add_argument('-beta1', type=float, default=0.9, help='First mement coefficient')
        self.parser.add_argument('-beta2', type=float, default=0.99, help='Second moment coefficient')
        self.parser.add_argument('-momentum', type=float, default=0, help='Momentum')

        # training options
        self.parser.add_argument('-num_epochs', type=int, default=100, help='Number of training epochs')
        self.parser.add_argument('-batch_size', type=int, default=256, help='Mini-batch size')
        self.parser.add_argument('-save_intervals', type=int, default=50, help='Number of iterations for saving model')
    
    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        
        # set directories for experiments
        self.opt.save_dir = '%s/%s_%s'%(conf.exp_dir,now.tm_mon, now.tm_mday)
        self.opt.save_dir = os.path.join(self.opt.save_dir, '%s'%(self.opt.idea))
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        # set directories for experiments
        '''
        _test == 'h36m':
            self.opt.save_dir = '%s/%s_%s'%(conf.exp_dir,now.tm_mon, now.tm_mday)
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s'%(self.opt.idea))
        elif self.opt.dataset_test == 'inf':
            self.opt.save_dir = '%s/%s_%s'%(mpi_exp,now.tm_mon, now.tm_mday)
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s'%(self.opt.idea))
        elif self.opt.dataset_test == 'rhd':
            self.opt.save_dir = '%s/%s_%s'%(hand_exp,now.tm_mon, now.tm_mday)
            self.opt.save_dir = os.path.join(self.opt.save_dir, '%s'%(self.opt.idea))
        '''
        #if self.opt.dataset_test == 'h36m':     
        #    self.opt.save_dir = '%s/test_%s_protocol%d' % (conf.exp_dir, self.opt.dataset_test, self.opt.protocol)
	
        self.opt.save_dir = os.path.join('%s/test_%s_protocol_%d'%(self.opt.save_dir, self.opt.dataset_test, self.opt.protocol))
        #if self.opt.dataset_test == 'h36m':     
        #    self.opt.save_dir = '%s/test_%s_protocol%d' % (conf.exp_dir, self.opt.dataset_test, self.opt.protocol)
        #elif self.opt.dataset_test == 'inf':
        #    self.opt.save_dir = '%s/test_%s' % (conf.exp_dir, self.opt.dataset_test)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        #self.opt.save_dir = os.path.join(self.opt.save_dir, '%s'%(self.opt.idea))
        #self.opt.save_dir = os.path.join(self.opt.save_dir, 'liftmesh')
        #if not os.path.exists(self.opt.save_dir):
        #    os.makedirs(self.opt.save_dir)
    

        if self.opt.noise != 1:
            if self.opt.scale == False:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_noise%d' % (self.opt.dataset_train, self.opt.noise))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_scale_noise%d' % (self.opt.dataset_train, self.opt.noise))
        else:
            if self.opt.scale == False:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_noise%d_std%.3f' % (self.opt.dataset_train, self.opt.noise, self.opt.std_train))
            else:
                self.opt.save_dir = os.path.join(self.opt.save_dir, 'train_%s_scale_noise%d_std%.3f' % (self.opt.dataset_train, self.opt.noise, self.opt.std_train))
        if self.opt.self == True:
            self.opt.save_dir = '%s_self' % (self.opt.save_dir)
        if self.opt.loop == True:
            self.opt.save_dir = '%s_loop' % (self.opt.save_dir)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        if self.opt.self == True:
            self.opt.save_dir = '%s/%1.1e_%1.1e_%1.1e_%1.1e_%1.1e_%s_lrg%1.1e_lrd%1.1e_batch%d' % \
                (self.opt.save_dir, self.opt.weight_2d,
                 self.opt.weight_adv_mesh, self.opt.weight_disc_mesh,
                 self.opt.weight_adv_2d, self.opt.weight_disc_2d,
                 self.opt.opt_method, self.opt.lr_g, self.opt.lr_d, self.opt.batch_size)
        else:
            self.opt.save_dir = '%s/%1.1e_%1.1e_%1.1e_%1.1e_%1.1e_%1.1e_%s_lrg%1.1e_lrd%1.1e_batch%d' % \
                (self.opt.save_dir, self.opt.weight_3d, self.opt.weight_2d,
                 self.opt.weight_adv_mesh, self.opt.weight_disc_mesh,
                 self.opt.weight_adv_2d, self.opt.weight_disc_2d,
                 self.opt.opt_method, self.opt.lr_g, self.opt.lr_d, self.opt.batch_size)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir, 0o777)

        # save options
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        refs = dict((name, getattr(conf, name)) for name in dir(conf)
                    if not name.startswith('_'))
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')
            for k, v in sorted(refs.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
                
        return self.opt

