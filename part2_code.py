import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

#####
def positional_encoding(x, num_frequencies=6, incl_input=True):
    
    """
    Apply positional encoding to the input.
    
    Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. 
    """
    
    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    
    for i in range(num_frequencies):
        for fn in [torch.sin, torch.cos]:
            results.append(fn((2.0 ** i) * np.pi * x))


    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)

######
def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ##########################  

    #y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))
    # x = x.float()
    # y = y.float()
    #normalization
    #pixel_coords = torch.stack([x[..., None], y[..., None], torch.ones(height, width, 1)], dim=-1)
    #cam_coords = torch.matmul(pixel_coords, torch.inverse(intrinsics))
    #cam_coords = F.normalize(cam_coords, dim=-1)

    #world_coords = torch.matmul(Rcw, cam_coords.unsqueeze(-1)) + Tcw.unsqueeze(-1)
    #world_coords = world_coords.squeeze(-1)

    #Rwc = Rcw.t()
    #Twc = -torch.matmul(Rwc, Tcw)
    #world_coords = torch.matmul(cam_coords, Rwc) + Twc
    #ray_directions = world_coords
    #ray_directions = torch.matmul(cam_coords, Rcw) #.t()
    #ray_directions = F.normalize(ray_directions, dim=-1)
    #ray_directions = torch.matmul(cam_coords, Rcw)
    #ray_directions = F.normalize(world_coords, dim=-1)
    #ray_origins = torch.reshape(Tcw, (1, 1, 3)).expand(height, width, 3
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))
    # print(y)
    # print(x)
    x = x.float()
    y = y.float()
    #normalization
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(height*width, 3).T
    # pixel_coords = torch.stack([x[..., None], y[..., None], torch.ones(height, width, 1)], dim=-1)
    #print(pixel_coords)
    #cam_coords = torch.matmul(pixel_coords, torch.inverse(intrinsics))
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords)
    # print(cam_coords)

    #ray_directions = torch.matmul(cam_coords, Rcw)
    ray_directions = torch.matmul(Rcw,cam_coords).T.reshape(height, width,3)
    #print(ray_directions)
    #ray_directions = F.normalize(ray_directions, dim=-1)
    ray_origins = torch.reshape(Tcw, (1, 1, 3)).expand(height, width, 3)
    #cam_center = torch.reshape(Twc, (1, 1, 3)).expand(height, width, 3)
    #ray_directions = world_coords - cam_center

    #ray_directions = torch.matmul(cam_coords, Rcw)
    #ray_directions = F.normalize(ray_directions, dim=-1)
    #ray_origins = torch.reshape(Tcw, (1, 1, 3)).expand(height, width, 3)
    #ray_origins = cam_center



    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################

    
    height, width, _ = ray_origins.shape
    
    t_sa = torch.rand(height, width, samples, device=ray_origins.device)
    t_sb = torch.arange(samples, device=ray_origins.device)
    t_samples = (t_sa+ t_sb ) / (samples + 1e-6)
    t_samples = t_samples.expand(height, width, samples)

    t_values = near * (1.0 - t_samples) + far * t_samples

    ray_points = ray_origins[..., None, :] + t_values[..., None] * ray_directions[..., None, :]
    depth_points = t_values

    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising 12 fully connected layers and following the 
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        self.layer1 = nn.Linear(3 + 2*3*num_x_frequencies, filter_size) #+ 3 * (num_d_frequencies*2 + 1)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, filter_size)
        self.layer4 = nn.Linear(filter_size, filter_size)
        self.layer5 = nn.Linear(filter_size, filter_size)

        self.layer6 = nn.Linear(filter_size + 3 + 2*3*num_x_frequencies, filter_size)
        self.layer7 = nn.Linear(filter_size, filter_size)
        self.layer8 = nn.Linear(filter_size, filter_size)

        self.layer9 = nn.Linear(filter_size, 1)
        self.layer10 = nn.Linear(filter_size, filter_size)
        self.layer11 = nn.Linear(filter_size + 3 + 2*3*num_d_frequencies, 128)
        self.layer12 = nn.Linear(128, 3)
        
        #self.sigma_layer = nn.Linear(filter_size, 1)
        #self.rgb_layer = nn.Linear(filter_size, 3)

        #self.num_x_frequencies = num_x_frequencies
        #self.num_d_frequencies = num_d_frequencies
        

        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        
        
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = F.relu(self.layer4(x3))
        x5 = F.relu(self.layer5(x4))

        x5com = torch.cat([x5, x], dim=-1)

        x6 = F.relu(self.layer6(x5com))

        x7 = F.relu(self.layer7(x6))
        x8 = F.relu(self.layer8(x7))

        sigma = self.layer9(x8)

        x10 = self.layer10(x8) # no relu here, it's a normal layer

        x10com = torch.cat([x10, d], dim=-1)

        x11 = F.relu(self.layer11(x10com))
        
        rgb = torch.sigmoid(self.layer12(x11))


        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
    
    ray_dir_den = torch.norm(ray_directions, dim=-1, keepdim=True)
    ray_directions_nrm = ray_directions / ray_dir_den

    ray_directions_pplt = ray_directions_nrm.unsqueeze(-2)
    ray_directions_pplt = ray_directions_pplt.repeat(1, 1, ray_points.shape[2], 1)

    ray_points_flt = ray_points.view(-1, 3)
    ray_directions_flt = ray_directions_pplt.view(-1, 3)

    
    peflt_ray_points = positional_encoding(ray_points_flt, num_frequencies=num_x_frequencies)
    peflt_ray_directions = positional_encoding(ray_directions_flt, num_frequencies=num_d_frequencies)

    #combined_data = torch.cat([pe_ray_points, pe_ray_directions], dim=-1)

    ray_points_batches = get_chunks(peflt_ray_points)
    ray_directions_batches = get_chunks(peflt_ray_directions)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################
    
    #weights = 1 - torch.exp(-s * (depth_points[1:] - depth_points[:-1]))
    #weights = 1 - torch.exp(-s * (depth_points[:, :, 1:] - depth_points[:, :, :-1]))
    #weights = weights * torch.cumprod(torch.cat([torch.ones((1, weights.shape[1], weights.shape[2]), device=weights.device), 1 - weights]), dim=0)
    #weights = 1 - torch.exp(-s * (depth_points[:, :, 1:] - depth_points[:, :, :-1]))
    #weights_padded = F.pad(weights, (0, 1), "constant", 1)
    #cumprod_weights = torch.cumprod(1 - weights_padded, dim=2)
    #weights_final = weights * cumprod_weights[:, :, :-1]
    # Apply the compositing weights to the RGB values to reconstruct the final image
    #rec_image = torch.sum(weights_final[..., None] * rgb, dim=2)


    
    device = rgb.device
    
    delta_depth = torch.ones_like(depth_points).to(device) * 1e9
    delta_depth[..., :-1] = torch.diff(depth_points, dim=-1)

    sigma_deltas = -F.relu(s) * delta_depth.reshape_as(s)
    T = torch.cumprod(torch.exp(sigma_deltas), dim=-1)
    T = torch.roll(T, shifts=1, dims=-1)

    C = (T * (1 - torch.exp(sigma_deltas)))[..., None] * rgb
    rec_image = C.sum(dim=-2)    






    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image
    #ray_origins, ray_directions = get_rays(height, width, intrinsics, pose[:3, :3], pose[:, 3])
    ray_origins, ray_directions = get_rays(height, width, intrinsics, pose[:3, :3], pose[:3, -1].reshape(-1,1))

    #sample the points from the rays
    #ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    #divide data into batches to avoid memory errors
    #ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)
    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    rgbs = []
    sigmas = []

    #forward pass the batches and concatenate the outputs at the end
    #for batch_points, batch_directions in zip(ray_points_batches, ray_directions_batches):
        #batch_rgb, batch_sigma = model(batch_points, batch_directions)
        #rgb_list.append(batch_rgb)
        #sigma_list.append(batch_sigma)
    del ray_origins, ray_directions, ray_points

    for x,d in zip(ray_points_batches, ray_directions_batches):
      rgb, sigma = model.forward(x,d)
      rgbs.append(rgb)
      sigmas.append(sigma)

    combined_rgb     = torch.cat(rgbs, dim=0).reshape(height, width, samples, -1)
    combined_sigma   = torch.cat(sigmas, dim=0).reshape(height, width, samples)
    #rgb = torch.cat(rgb_list, dim=2)
    #sigma = torch.cat(sigma_list, dim=2)

    # Apply volumetric rendering to obtain the reconstructed image
    #rec_image = volumetric_rendering(rgb, sigma, depth_points)
    rec_image = volumetric_rendering(combined_rgb, combined_sigma, depth_points)

    #############################  TODO 2.5 END  ############################

    return rec_image