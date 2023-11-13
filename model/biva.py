
#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional
# from collections import defaultdict
from torch.distributions import Normal
import logging
import wandb
import os
import datetime
import sys
sys.path.append("..")

from model.blocks import BlockEncoder, AsFeatureMap_down, AsFeatureMap_up, BlockDecoder, BlockFinalImg, Upsample, BlockQ
from model.utils import soft_clamp, soft_clamp_img, InitWeights_He, SEBlock3D
from model.classifier import MLP

def create_logger(folder):
    """Create a logger to save logs."""
    compt = 0
    while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
        compt+=1
    logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(WandbHandler())

    return logger 


class WandbHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        wandb.run.log({record.levelname: msg})
        

class BIVA3D(nn.Module):

    def __init__(
        self, 
        stochastic_module,
        resume,
        root,
        base_num_features, 
        num_pool,
        expansion_factor=2,
        num_feat_img=1,
        init_weights=InitWeights_He,
        input_shape=(64,64,64),
        max_features=32,
        with_residual=True,
        with_se=True,
        logger=None,
        classifier=MLP(4096),
        last_act='tanh'):
        """

        """
        super(BIVA3D, self).__init__()

        self.init_weights = init_weights        
        self.current_img_dims = input_shape
        
        self.base_num_features = base_num_features
        self.max_features = max_features
        self.num_feat_img = num_feat_img
        self.num_pool = num_pool
        
        self.classifier = classifier
        
        self.resume = resume
        
        self.last_act = last_act
        self.with_residual = with_residual
        self.with_se = with_se

        self.stochastic_module = stochastic_module
        
        self.expansion_factor = expansion_factor

        self.build_init()
        
        
        if self.resume:
            self.root = root
            # Create a directory with the timestamp
            save_dir = os.path.join(self.root, "models")
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

            log_dir = os.path.join(self.root, "logs")
            os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist            
            
        else:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            self.root = os.path.join("/root/code/pg_hvae/runs",timestamp)

            # Create a directory with the timestamp
            save_dir = os.path.join(self.root, "models")
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

            log_dir = os.path.join(self.root, "logs")
            os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

        self.save_dir = save_dir
        self.logger = create_logger(log_dir)

        
    def build_init(self):

        # min_shape = [int(k/(2**self.num_pool)) for k in self.original_shape] # Define shape of the smallest feature volume
        min_shape = [int(k/(2**self.num_pool)) for k in self.current_img_dims] # Define shape of the smallest feature volume

        down_strides = [(2,2,2)]*(self.num_pool) # Define strides for downsampling
        up_kernel = down_strides

        nfeat_input = 1 
        nfeat_output = self.base_num_features

        self.first_conv = nn.Conv3d(
                        in_channels=nfeat_input, 
                        out_channels=nfeat_output,
                        kernel_size=3,
                        stride=1, 
                        padding=1,
                        bias=True)

        self.conv_blocks = []
        self.td = []
        
        # TODO make correspond to progressive grow stage
        for d in range(self.num_pool): # Initial pooling step (1 or 2)
            
            nfeat_input = nfeat_output
            
            nfeat_output = min(2 * nfeat_output, self.max_features)
            self.nfeat_output = nfeat_output

            self.conv_blocks.append(BlockEncoder(nfeat_input, nfeat_input, residual=self.with_residual, with_se=self.with_se))
            
            self.td.append(nn.Conv3d(nfeat_input, nfeat_output, kernel_size=3, stride=down_strides[d], padding=1))
            
        self.conv_blocks.append(BlockEncoder(nfeat_output, nfeat_output, residual=self.with_residual, with_se=self.with_se))

        # Going from a feature volume to a feature vector (e.g (3,3,3)-->(1))
        self.bottleneck_down = AsFeatureMap_down(input_shape=[nfeat_output,]+min_shape, target_dim=2*self.max_features)
        # Going from a feature vector to a feature volume (e.g. (1)-->(3,3,3)) 
        self.bottleneck_up = AsFeatureMap_up(input_dim=1*self.max_features, target_shape=[self.max_features,]+min_shape)

        # Layers for the approximate posterior + prior
        nfeat_latent = self.max_features
                
        self.tu = []
        self.conv_blocks_localization = []
        self.qz = []
        self.pz = []
        
        for u in np.arange(self.num_pool)[::-1]:
            
            nfeatures_from_skip = self.conv_blocks[u].output_channels            
            n_features_after_tu_and_concat = 2*nfeatures_from_skip
            
            self.tu.append(
                    Upsample(n_channels=nfeat_latent, n_out=nfeatures_from_skip, scale_factor=up_kernel[u], mode='trilinear')
                    )
            
            self.conv_blocks_localization.append(BlockDecoder(nfeatures_from_skip, residual=self.with_residual, with_se=self.with_se))

            self.qz.append(BlockQ(n_features_after_tu_and_concat, nfeatures_from_skip, nfeatures_from_skip))
            self.pz.append(nn.utils.weight_norm(nn.Conv3d(nfeatures_from_skip, nfeatures_from_skip, 1, 1, 0, 1, 1, False), dim=0, name='weight'))

            nfeat_latent = nfeatures_from_skip // 2
            self.nfeat_latent = nfeat_latent
            
        # TO DO - change from [final_block] to final_block (don't need list or self.return_cat)
        self.final_blocks = BlockFinalImg(nfeat_latent, self.num_feat_img, self.last_act)
        
        self.nb_latent = len(self.conv_blocks) # Get number of latent layers from number of conv blocks
        
        if self.init_weights is not None:
            self.apply(self.init_weights)
            
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        self.td = nn.ModuleList(self.td)
        
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)

        self.qz = nn.ModuleList(self.qz)
        self.pz = nn.ModuleList(self.pz)


    def create_encodings(self, x):
        # Encode x
        skips = []
        x = self.first_conv(x)
        for d in range(len(self.conv_blocks) - 1): #self.nb_latent - 1): # TD pass
            x = self.conv_blocks[d](x)
            skips.append(x)
            x = self.td[d](x)
        
        x = self.conv_blocks[-1](x)
        return x, skips
    
    def update_train_stage(self):
        self.train_stage += 1
        self.logger.info(f"Train stage: {self.train_stage}")
        return self.train_stage
    
    # TODO move to stochastic module
    def compute_marginal(self, mu_q_res, log_sigma_q_res, mu_p, inv_sigma_p, temp=1.0):
        inv_sigma_q_res = torch.exp(-log_sigma_q_res)
        sigma_q = 1 / (inv_sigma_q_res + inv_sigma_p + 1e-3)
        mu_q = sigma_q * (inv_sigma_q_res * mu_q_res + inv_sigma_p * mu_p)

        return Normal(mu_q, temp*sigma_q)
    
    # TODO move to stochastic module
    def compute_full(self, res_params, prior, temp):
        mu = prior.loc / prior.scale
        inv_sigma = 1 / prior.scale
        
        mu+= res_params['loc'] * torch.exp(-res_params['logscale_res'])
        inv_sigma += torch.exp(-res_params['logscale_res'])
        mu /= inv_sigma
        sigma = 1 / inv_sigma

        return Normal(mu, temp*sigma)
        

    def forward(self, x, temp=1, return_feat=False, return_z=True, verbose=False):
        
        # Create embeddings for injecting info from (x_i)
        x, skips = self.create_encodings(x)
        
        # TODO move to stochastic module
        # Initialization of distributions and their parameters (for every level)
        distribs = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        res_params = {f'z{i+1}':dict() for i in range(self.nb_latent)}

        # TODO move to stochastic module 
        # Computing q(z_L|x_i) 
        z_name = 'z{}'.format(self.nb_latent)

        mu_zl_q_res, logvar_zl_q_res = self.bottleneck_down(x).chunk(2, dim=1) # TODO put chunk in stochastic module

        # TODO move to stochastic module - Constrain mu and logvar. Put soft clamp/transforms to stochastic utils
        mu_zl_q_res = soft_clamp(mu_zl_q_res)
        logvar_zl_q_res = soft_clamp(logvar_zl_q_res)

        # Populate res_params and distribs dicts
        res_params[z_name] = {'loc':mu_zl_q_res, 'logscale_res': logvar_zl_q_res}
        distribs[z_name]['marginal'] = self.compute_marginal(mu_zl_q_res, logvar_zl_q_res, 0, 1, temp) # prior is N(0,I)
        
        # p(z_L) - define prior to z_L
        mu_zl_p = torch.zeros_like(mu_zl_q_res)
        sigma_zl_p = torch.ones_like(logvar_zl_q_res)

        distribs[z_name]['prior'] = Normal(mu_zl_p, sigma_zl_p)
        # distribs[z_name].update('prior'=Normal(mu_zl_p, sigma_zl_p))

        # Approximate posterior for q(z_L|x_{\pi}) = p(z_L) \prod_{i\in\pi} q(z_L|x_i)
        distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
        # Sampling zL_q from q(z_L|x_{\pi})
        zl_q = distribs[z_name]['full'].rsample()
        if verbose:
            self.logger.info(f"Shape {z_name}: {zl_q.size()}")
        
        # Computing KLs
        kls = dict()
        kls['prior'] = []
        kl = distribs[z_name]['full'].log_prob(zl_q) - distribs[z_name]['marginal'].log_prob(zl_q)
        kls['prior'].append(kl.sum())
        
        # Creating initial feature volume for z_{L-1}
        zl_q_up = self.bottleneck_up(zl_q)
        z_full = {z_name:zl_q_up}

        for i in range(self.nb_latent - 1): 
            z_name = 'z{}'.format(self.nb_latent-(i+1)) # = z^{l-1}
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)] # get z_l from dict
            x = self.tu[i](z_ip1) # do tu pass
            x = self.conv_blocks_localization[i](x) # do conv pass
            if verbose:
                self.logger.info(f"Shape feature volume for {z_name}: {x.size()}")
            
            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p)
            logvar_zi_p = soft_clamp(logvar_zi_p)
            distribs[z_name]['prior'] = Normal(mu_zi_p, torch.exp(logvar_zi_p))
            
            # Computing  q(z_{l-1}|x_i,z_l) 

            # Merging embedding from z_{l-1} and x_i
            x_q = torch.cat((x, skips[-(i + 1)]), dim=1)
            mu_zi_q_res, logvar_zi_q_res = self.qz[i](x_q).chunk(2, dim=1)
            mu_zi_q_res = soft_clamp(mu_zi_q_res)
            logvar_zi_q_res = soft_clamp(logvar_zi_q_res)
            res_params[z_name] = {'loc':mu_zi_q_res, 'logscale_res': logvar_zi_q_res}
            distribs[z_name]['marginal'] = self.compute_marginal(mu_zi_q_res, logvar_zi_q_res, mu_zi_p, torch.exp(-logvar_zi_p), temp) 

            # Approximate posterior for q(z_{l-1}|x_{\pi}, z_l) = p(z_{l-1}|z_l) \prod_{i\in\pi} q(z_{l-1}|x_i,z_l)
            distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
            # Sampling z_{l-1}
            zi_q = distribs[z_name]['full'].rsample()
            if verbose:
                self.logger.info(f"Shape {z_name}: {zi_q.size()}")

            # Computing KLs
            kl = distribs[z_name]['full'].log_prob(zi_q) - distribs[z_name]['prior'].log_prob(zi_q)
            kls['prior'].append(kl.sum())
            z_full[z_name] = zi_q            

        output_img = self.final_blocks(z_full['z1'])

        if return_feat:
            return  output_img, kls, z_full['z1']
        elif return_z:
            return output_img, kls, z_full['z{}'.format(self.nb_latent)]
        else:
            return  output_img, kls        
    

    def sample(self, batch_size, temp=0.7):
        
        # Prior distribution for z_L
        mu = torch.zeros((batch_size,self.max_features)).cuda()
        sigma = torch.ones((batch_size,self.max_features)).cuda()

        p_zl = Normal(mu, temp*sigma) # define prior to z_L
        
        # Sample from p(z_L)
        zl_p = p_zl.sample() # prior sample
        zl_p_up = self.bottleneck_up(zl_p) # 

        z_full = {'z{}'.format(self.nb_latent):zl_p_up} # dict of latent variables from prior

        for i in range(self.nb_latent - 1):
            z_name = 'z{}'.format(self.nb_latent-(i+1))
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)

            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p)
            logvar_zi_p = soft_clamp(logvar_zi_p)
            var_zi_p = torch.exp(logvar_zi_p)
            p_zi =  Normal(mu_zi_p, temp*var_zi_p)
            
            # Sampling z_{l-1}
            zi_p = p_zi.sample()
            z_full[z_name] = zi_p

        # if self.return_cat:
        #     return torch.cat([fblock(z_full['z1']) for fblock in self.final_blocks], 1)
        # else:
        return self.final_blocks(z_full['z1'])

    def dir_sample(self, shape):
        alphas = torch.full((shape), self.dir_alphas).double().cuda() # sparse prior
        p_zl = Dirichlet(alphas, validate_args=True)

        zl_p = p_zl.sample()
        zl_p_up = self.bottleneck_up(zl_p)
        z_full = {'z{}'.format(self.nb_latent):zl_p_up} # dict of latent variables from prior

        for i in range(self.nb_latent - 1):
            z_name = 'z{}'.format(self.nb_latent-(i+1))
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)

            # Prior p(z_{l-1}|z_l)
            alphas_zi_p = self.pz[i](x)
            alphas_zi_p = F.softplus(alphas_zi_p) + 1e-10

            p_zi =  Dirichlet(alphas_zi_p) 
            
            # Sampling z_{l-1}
            zi_p = p_zi.sample()
            z_full[z_name] = zi_p

        # if self.return_cat:
        #     return torch.cat([fblock(z_full['z1']) for fblock in self.final_blocks], 1)
        # else:
        return self.final_blocks(z_full['z1'])

    def gau_sample(self, batch_size, temp=0.7):
        
        # Prior distribution for z_L
        mu = torch.zeros((batch_size,2*self.max_features)).cuda()
        sigma = torch.ones((batch_size,2*self.max_features)).cuda()

        p_zl = Normal(mu, temp*sigma) # define prior to z_L
        
        # Sample from p(z_L)
        zl_p = p_zl.sample() # prior sample
        zl_p_up = self.bottleneck_up(zl_p) # 

        z_full = {'z{}'.format(self.nb_latent):zl_p_up} # dict of latent variables from prior

        for i in range(self.nb_latent - 1):
            z_name = 'z{}'.format(self.nb_latent-(i+1))
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)

            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p)
            logvar_zi_p = soft_clamp(logvar_zi_p)
            var_zi_p = torch.exp(logvar_zi_p)
            p_zi =  Normal(mu_zi_p, temp*var_zi_p)
            
            # Sampling z_{l-1}
            zi_p = p_zi.sample()
            z_full[z_name] = zi_p

        # if self.return_cat:
        #     return torch.cat([fblock(z_full['z1']) for fblock in self.final_blocks], 1)
        # else:
        return self.final_blocks(z_full['z1'])

    def model_save(self, train_stage):
        path = f"{self.save_dir}/model{train_stage}.pth"
        torch.save(self.state_dict(), path)

    def model_load(self, stage):
        # path = f"{path}/model{self.train_stage - 1}.pth"
        full_path = f"{self.save_dir}/model{stage}-klweight-4.pth"
        self.load_state_dict(torch.load(full_path))
    
    def find_input_dims(self):
        params = list(self.named_parameters())
        return params[0][1].size()

    def get_new_dims(self):
        new_img_dims = [i*self.expansion_factor for i in self.current_img_dims]
        return new_img_dims
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            

    def add_layers(self):
        
        min_shape = [int(k/(2**self.num_pool)) for k in self.current_img_dims] # Define shape of the smallest feature volume

        num_pool = 1 #self.new_img_dims[2] // self.current_img_dims[2] - 1
            
        down_strides = [(2,2,2)]*(num_pool) # Define strides for downsampling

        up_kernel = down_strides
            
        print(f"Next stage - adding {num_pool} layers")
        print(f"Applying downstrides of - {down_strides}")
        print(f"With base number of features - {self.base_num_features}")

        nfeat_input = 1 
        nfeat_output = self.base_num_features

        self.new_conv_blocks = []
        self.new_td = []
        
        # TODO make correspond to progressive grow stage
        for d in range(num_pool): # Initial pooling step (1 or 2)
            
            nfeat_input = nfeat_output
            
            self.new_conv_blocks.append(BlockEncoder(nfeat_input, nfeat_input, residual=self.with_residual, with_se=self.with_se))
            self.new_td.append(nn.Conv3d(nfeat_input, nfeat_output, kernel_size=3, stride=down_strides[d], padding=1))
            
            nfeat_output = min(2 * nfeat_output, self.max_features)

        self.conv_blocks = nn.ModuleList(self.new_conv_blocks + list(self.conv_blocks))
        self.td = nn.ModuleList(self.new_td + list(self.td)) 
        
        self.nb_latent = len(self.conv_blocks) # Get number of latent layers from number of conv blocks
        nfeat_latent = self.nfeat_latent

        self.new_tu = []
        self.new_conv_blocks_localization = []
        self.new_qz = []
        self.new_pz = []


        for u in np.arange(num_pool)[::-1]:
            
            nfeatures_from_skip = list(self.new_conv_blocks)[u].output_channels            
            n_features_after_tu_and_concat = 2*nfeatures_from_skip 
                                            
            self.new_tu.append(
                    Upsample(n_channels=nfeat_latent, n_out=nfeatures_from_skip, scale_factor=up_kernel[u], mode='trilinear')
                    )
            
            self.new_conv_blocks_localization.append(BlockDecoder(nfeatures_from_skip, residual=self.with_residual, with_se=self.with_se))

            self.new_qz.append(BlockQ(n_features_after_tu_and_concat, nfeatures_from_skip, nfeatures_from_skip))
            self.new_pz.append(nn.utils.weight_norm(nn.Conv3d(nfeatures_from_skip, nfeatures_from_skip, 1, 1, 0, 1, 1, False), dim=0, name='weight'))

            nfeat_latent = nfeatures_from_skip // 2
            
#         # TO DO - change from [final_block] to final_block (don't need list or self.return_cat)
#         self.final_blocks = BlockFinalImg(nfeat_latent, self.num_feat_img, self.last_act)
        
        self.tu = nn.ModuleList(list(self.tu) + self.new_tu)            
        self.conv_blocks_localization = nn.ModuleList(list(self.conv_blocks_localization) + self.new_conv_blocks_localization)
        self.pz = nn.ModuleList(list(self.pz) + self.new_pz)
        self.qz = nn.ModuleList(list(self.qz) + self.new_qz)
        
        self.num_pool = num_pool
        self.nfeat_latent = nfeat_latent
        
    def grow(self, stage):
        
        self.freeze()
        self.new_img_dims = self.get_new_dims()
        
        assert self.new_img_dims[0] == self.new_img_dims[1] == self.new_img_dims[2]
        
        if self.new_img_dims[2] % self.current_img_dims[2] !=0:
            raise ValueError("New image dims not divisible by current image dims")

        print("Expanding model...")
        print(f"Current image dimensions: {self.current_img_dims}")
        print(f"Expansion factor: {self.expansion_factor}")
        print(f"New image dimensions: {self.new_img_dims}")
        
        # Load weights into the previous stage
        
        previous_path = f"{self.save_dir}/model{stage-1}.pth"
        prev_model = torch.load(previous_path)
        self.load_state_dict(prev_model)
        
        self.add_layers()
        
        self.current_img_dims = self.new_img_dims
        
               
        

if __name__ == "__main__":
    model = BIVA3D(stochastic_module=None, base_num_features=32, num_pool=2)

    x = torch.randn(1, 1, 32, 32, 32)
    output_img, kls = model(x)
    
    assert x.shape == output_img.shape
    
    model.grow(2)
    x = torch.randn(1, 1, 64, 64, 64)
    output_img, kls = model(x)
    
    print("============")
   
    
    model.grow(2)
    x = torch.randn(1, 1, 128, 128, 128)
    output_img, kls = model(x)
    print("output", output_img.shape)

    model.grow(3)
    x = torch.randn(1, 1, 256, 256, 256)
    output_img, kls = model(x)
    print("output", output_img.shape)