import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from collections import OrderedDict
import cv2

# import torch.nn as nn
torch.cuda.set_device(3)


def ray_sampling(Ks, Ts, image_size, masks=None, mask_threshold=0.5, images=None):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)

    x = torch.linspace(0, h - 1, steps=h, device=Ks.device)
    y = torch.linspace(0, w - 1, steps=w, device=Ks.device)

    grid_x, grid_y = torch.meshgrid(x, y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M, 1, 1, 1)  # (M,2,H,W)
    coordinates = torch.cat([coordinates, torch.ones(coordinates.size(0), 1, coordinates.size(2),
                                                     coordinates.size(3), device=Ks.device)], dim=1).permute(0, 2, 3,
                                                                                                             1).unsqueeze(
        -1)

    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks, coordinates)  # (M,H,W,3,1)
    dirs = dirs / torch.norm(dirs, dim=3, keepdim=True)
    dirs = torch.cat([dirs, torch.zeros(dirs.size(0), coordinates.size(1),
                                        coordinates.size(2), 1, 1, device=Ks.device)], dim=3)  # (M,H,W,4,1)

    dirs = torch.matmul(Ts, dirs)  # (M,H,W,4,1)
    dirs = dirs[:, :, :, 0:3, 0]  # (M,H,W,3)

    pos = Ts[:, 0:3, 3]  # (M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)

    rays = torch.cat([dirs, pos], dim=3)  # (M,H,W,6)

    if images is not None:
        rgbs = images.permute(0, 2, 3, 1)  # (M,H,W,C)
    else:
        rgbs = None

    if masks is not None:
        rays = rays[masks > mask_threshold, :]
        if rgbs is not None:
            rgbs = rgbs[masks > mask_threshold, :]

    else:
        rays = rays.reshape((-1, rays.size(3)))
        if rgbs is not None:
            rgbs = rgbs.reshape((-1, rgbs.size(3)))

    return rays, rgbs


class RaySamplePoint_Near_Far(nn.Module):
    def __init__(self, sample_num=64):
        super(RaySamplePoint_Near_Far, self).__init__()
        self.sample_num = sample_num

    def forward(self, rays, near_far):
        '''
        :param rays: N*6
        :param bbox: N*8*3  0,1,2,3 bottom 4,5,6,7 up
        pdf: n*coarse_num 表示权重
        :param method:
        :return: N*C*3
        '''
        n = rays.size(0)

        ray_d = rays[:, :3]
        ray_o = rays[:, 3:]

        t_vals = torch.linspace(0., 1., steps=self.sample_num, device=rays.device)
        # print(near_far[:,0:1].repeat(1, self.sample_num).size(), t_vals.unsqueeze(0).repeat(n,1).size())
        z_vals = near_far[:, 0:1].repeat(1, self.sample_num) * (1. - t_vals).unsqueeze(0).repeat(n, 1) + near_far[:,
                                                                                                         1:2].repeat(1,
                                                                                                                     self.sample_num) * (
                     t_vals.unsqueeze(0).repeat(n, 1))

        # z_vals = near * (1.-t_vals) +  far  * (t_vals)
        # z_vals = z_vals.expand([n, self.sample_num])

        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = torch.rand(z_vals.size(), device=rays.device)

        z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]

        return z_vals.unsqueeze(-1), pts


def gen_weight(sigma, delta, act_fn=F.relu, expscale=1.0):
    """Generate transmittance from predicted density
    """
    alpha = 1. - torch.exp(-expscale * act_fn(sigma.squeeze(-1)) * delta)
    weight = 1. - alpha + 1e-10
    # weight = alpha * torch.cumprod(weight, dim=-1) / weight # exclusive cum_prod

    weight = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), weight], -1), -1)[:,
                     :-1]

    return weight


class VolumeRenderer(nn.Module):
    def __init__(self, use_mask=False, boarder_weight=1e10, expscale=1.0):
        super(VolumeRenderer, self).__init__()
        self.boarder_weight = boarder_weight
        self.use_mask = use_mask
        self.expscale = expscale

    def forward(self, depth, rgb, sigma, rays=None, noise=0):
        """
        N - num rays; L - num samples;
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]

        :return:
            color: torch.tensor [N, 3]
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        # pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
        pad = self.boarder_weight * torch.ones(delta[..., :1].size(), device=delta.device)
        delta = torch.cat([delta, pad], dim=-1)  # [N, L]

        # if rays is not None:
        #    delta = delta * torch.norm(rays[...,0:3], dim=1,keepdim =True).repeat(1,delta.size(1))

        if noise > 0.:
            sigma += (torch.randn(size=sigma.size(), device=delta.device) * noise)

        weights = gen_weight(sigma, delta, expscale=self.expscale).unsqueeze(-1)  # [N, L, 1]

        color = torch.sum(torch.sigmoid(rgb) * weights, dim=1)  # [N, 3]
        depth = torch.sum(weights * depth, dim=1)  # [N, 1]
        acc_map = torch.sum(weights, dim=1)  #

        if self.use_mask:
            color = color + (1. - acc_map[..., None])

        return color, depth, acc_map, weights


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding
class Trigonometric_kernel:
    def __init__(self, L=10):
        self.L = L

        self.embed_fn, self.out_ch = get_embedder(L)

    '''
    INPUT
     x: input vectors (N,C) 

     OUTPUT

     pos_kernel: (N, calc_dim(C) )
    '''

    def __call__(self, x):
        return self.embed_fn(x)

    def calc_dim(self, dims=0):
        return self.out_ch


class SpaceNet(nn.Module):

    def __init__(self, c_pos=3, include_input=True):
        super(SpaceNet, self).__init__()

        self.tri_kernel_pos = Trigonometric_kernel(L=10)
        self.tri_kernel_dir = Trigonometric_kernel(L=4)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)

        backbone_dim = 256
        head_dim = 128

        self.placeholder_quant = nn.Identity()

        l = OrderedDict()
        for i in range(4):
            if i == 0:
                l['%d_stage1' % i] = nn.Linear(self.pos_dim, backbone_dim)
            else:
                l['%d_stage1' % i] = (nn.Linear(backbone_dim, backbone_dim))
            l['%d_stage1_relu' % i] = (nn.ReLU(inplace=True))

        self.stage1 = nn.Sequential(l)

        l = OrderedDict()

        for i in range(3):
            if i == 0:
                l['%d_stage2' % i] = (nn.Linear(backbone_dim + self.pos_dim, backbone_dim))
            else:
                l['%d_stage2_relu' % i] = (nn.ReLU(inplace=True))
                l['%d_stage2' % i] = (nn.Linear(backbone_dim, backbone_dim))

        self.stage2 = nn.Sequential(l)

        l = OrderedDict()
        pres = backbone_dim
        for i in range(1):
            l['%d_density_relu' % i] = (nn.ReLU(inplace=True))
            if i == 0:
                l['%d_density' % i] = (nn.Linear(pres, 1))
            else:
                l['%d_density' % i] = (nn.Linear(pres, head_dim))
                pres = head_dim

        self.density_net = nn.Sequential(l)

        l = OrderedDict()
        pres = backbone_dim + self.dir_dim
        for i in range(2):
            l['%d_rgb_relu' % i] = (nn.ReLU(inplace=True))
            if i == 1:
                l['%d_rgb' % i] = (nn.Linear(pres, 3))
            else:
                l['%d_rgb' % i] = (nn.Linear(pres, head_dim))
                pres = head_dim

        self.rgb_net = nn.Sequential(l)

    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    '''

    def forward(self, pos, rays):

        # beg = time.time()
        rgbs = None
        if rays is not None:
            dirs = rays[..., 0:3]

        bins_mode = False
        if len(pos.size()) > 2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1, self.c_pos))  # (N,c_pos)
            if rays is not None:
                dirs = dirs.unsqueeze(1).repeat(1, L, 1)
                dirs = dirs.reshape((-1, self.c_pos))  # (N,3)

        pos = self.placeholder_quant(pos)

        pos = self.tri_kernel_pos(pos)
        pos = self.placeholder_quant(pos)
        if rays is not None:
            dirs = self.placeholder_quant(dirs)
            dirs = self.tri_kernel_dir(dirs)
            dirs = self.placeholder_quant(dirs)

        # torch.cuda.synchronize()
        # print('transform :',time.time()-beg)

        # beg = time.time()
        x = self.stage1(pos)
        x = self.stage2(torch.cat([x, pos], dim=1))

        density = self.density_net(x)

        if rays is not None:
            rgbs = self.rgb_net(torch.cat([x, dirs], dim=1))

        # torch.cuda.synchronize()
        # print('fc:',time.time()-beg)

        if bins_mode:
            density = density.reshape((-1, L, 1))
            if rays is not None:
                rgbs = rgbs.reshape((-1, L, 3))

        return rgbs, density


if __name__ == '__main__':

    camera_info = torch.load('./data/dump_data_input_camera.pth', map_location='cpu')

    K = camera_info['K']  # intrinsic
    T = camera_info['T']  # extrinsic
    image_size = camera_info['image_size']

    '''
       Ray_reconstruction Test
       --------------------------------------------
       Input: K,T,image_size
       Output: rays_generated
    '''
    rays_generated, _ = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), image_size)

    rays_list = rays_generated.split(1024 * 7, dim=0)

    sampler = RaySamplePoint_Near_Far(sample_num=16)
    print(len(rays_list))

    print(rays_generated.size())

    # build a network
    net = SpaceNet().cuda()

    # load trained model
    net.load_state_dict(torch.load('spacenet.pth', map_location='cpu'))

    volume_render = VolumeRenderer(boarder_weight=1e-10, expscale=1.0)

    colors = []
    depths = []
    acc_maps = []

    sum_time = 0

    for i in range(67):
        res = torch.load('./data/dump_data_spacenet_%d.pth' % i, map_location='cpu')

        # pos_gt = res['xyz'].cuda()
        # rays_gt = res['rays'].cuda()
        rays = rays_list[i].cuda()

        near = res['near'].cuda()
        far = res['far'].cuda()

        ori_size = res['ori_size']
        sampled_rays_coarse_t_gt = res['t'].cuda()
        ray_mask = res['ray_mask']

        near = near[ray_mask, :]
        far = far[ray_mask, :]

        rays = rays[ray_mask, :]

        '''
           Ray_sampling Test
           --------------------------------------------
           Input: rays, near_far
           Output: sampled_rays_coarse_t, pos
        '''
        sampled_rays_coarse_t, pos = sampler(rays, torch.cat([near, far], dim=1))

        torch.cuda.synchronize()
        s_time = time.time()

        '''
           Network Test
           --------------------------------------------

        '''
        # execute network
        with torch.no_grad():
            rgbs, density = net(pos, rays)

        torch.cuda.synchronize()
        sum_time = sum_time + (time.time() - s_time)

        density[sampled_rays_coarse_t[:, :, 0] < 0, :] = 0.0

        '''
           Volume rendering test
           --------------------------------------------

        '''
        color_0, depth_0, acc_map_0, weights_0 = volume_render(sampled_rays_coarse_t, rgbs, density, rays, noise=0.0)

        color_final_0 = torch.zeros(ori_size, 3, device=rays.device)
        color_final_0[ray_mask] = color_0
        depth_final_0 = torch.zeros(ori_size, 1, device=rays.device)
        depth_final_0[ray_mask] = depth_0
        acc_map_final_0 = torch.zeros(ori_size, 1, device=rays.device)
        acc_map_final_0[ray_mask] = acc_map_0

        colors.append(color_final_0)
        depths.append(depth_final_0)
        acc_maps.append(acc_map_final_0)

        print(color_final_0.size())

    colors = torch.cat(colors, dim=0)
    depths = torch.cat(depths, dim=0)
    acc_maps = torch.cat(acc_maps, dim=0)

    colors = colors.reshape(600, 800, 3).detach().cpu().numpy()
    colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
    print(colors.shape)

    print("total time:", sum_time)

    # output image
    cv2.imwrite('img.jpg', colors * 255.0)


