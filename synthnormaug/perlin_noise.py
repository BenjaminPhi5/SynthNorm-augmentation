import torch
import numpy as np

"""
perlins improved noise: based on the following algorithm:
https://dl.acm.org/doi/pdf/10.1145/566654.566636
Perlin, Ken. "Improving noise." Proceedings of the 29th annual conference on Computer graphics and interactive techniques. 2002.
"""

G = torch.tensor([ # series of vectors to sample from randomly in perlin's improved noise algorithm.
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ]
)

# coordinates of each corner on the unit cube.
cube_corner_coords = torch.tensor([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
]).unsqueeze(1)

def Smooth(t):
    return 6*t**5 - 15*t**4 + 10*t**3

def perlin_improved_noise_gpu(shape=[80, 192, 160], res=[160, 192, 160], device='cuda'):

    s = torch.tensor(shape, device=device)
    res = torch.tensor(res, device=device)

    grid = torch.stack(torch.meshgrid(
            torch.arange(0, s[0], device=device),
            torch.arange(0, s[1], device=device),
            torch.arange(0, s[2], device=device),
            indexing='ij'
        ), dim=-1)
    grid = grid.float() / res
    flat_grid = grid.reshape(-1, 3)
    
    # vec grid v2
    vec_grid_shape = torch.tensor((int(s[0] / res[0]) + 2, int(s[1] / res[1]) + 2, int(s[2] / res[2]) + 2), dtype=torch.long)
    
    P = torch.randperm(torch.prod(vec_grid_shape)) % 12
    vec_grid = G[P].reshape([*vec_grid_shape, 3]).to(device)

    # bounding box coordinates for each point in the grid
    lower_left_loc = torch.floor(flat_grid).type(torch.int32)
    corner_locs = lower_left_loc.unsqueeze(0) + cube_corner_coords.to(device)

    # compute dot products
    def grid_loc_dot(flat_grid, vec_grid, loc):
        floc = loc.reshape(-1, 3)
        vec_grid_vals = vec_grid[floc[:,0], floc[:,1], floc[:,2]].reshape(loc.shape)
        return (-(flat_grid - loc) * vec_grid_vals).sum(dim=-1)

    dot_prods = grid_loc_dot(flat_grid, vec_grid, corner_locs)

    # apply smoothing function to distance to corner
    smoothed_diff = Smooth(flat_grid - lower_left_loc)

    # apply trilinear interpolation
    s000, s001, s010, s011, s100, s101, s110, s111 = dot_prods
    x00 = s000 + (smoothed_diff[:,0]) * (s100 - s000)
    x01 = s001 + (smoothed_diff[:,0]) * (s101 - s001)
    x10 = s010 + (smoothed_diff[:,0]) * (s110 - s010)
    x11 = s011 + (smoothed_diff[:,0]) * (s111 - s011)
    
    y0 = x00 + (smoothed_diff[:,1]) * (x10 - x00)
    y1 = x01 + (smoothed_diff[:,1]) * (x11 - x01)
    
    z = y0 + (smoothed_diff[:,2]) * (y1 - y0)
    R = z.reshape(grid.shape[0:3])
    
    return R

def perlin_improved_noise(shape=[80, 192, 160], res=[160, 192, 160]):

    device = 'cpu'
    s = shape
    res = np.array(res)


    grid = torch.stack(torch.meshgrid(
            torch.arange(0, s[0], device=device),
            torch.arange(0, s[1], device=device),
            torch.arange(0, s[2], device=device),
            indexing='ij'
        ), dim=-1)
    
    grid = grid.float() / res
    flat_grid = grid.reshape(-1, 3)
    
    # vec grid v2
    vec_grid_shape = torch.tensor((int(s[0] / res[0]) + 2, int(s[1] / res[1]) + 2, int(s[2] / res[2]) + 2), dtype=torch.long)
    # print(vec_grid_shape)
    P = torch.randperm(torch.prod(vec_grid_shape)) % 12
    vec_grid = G[P].reshape([*vec_grid_shape, 3]).to(device)
    
    
    # # vec grid v0
    # vec_grid = torch.stack(torch.meshgrid(
    #         torch.arange(0, s[0]/res[0] + 1, device=device),
    #         torch.arange(0, s[1]/res[1] + 1, device=device),
    #         torch.arange(0, s[2]/res[2] + 1, device=device),
    #         indexing='ij'
    #     ), dim=-1)
    
    # vec_grid[:,:,:,0] = torch.rand(*vec_grid[:,:,:,0].shape) * 2 - 1
    # vec_grid[:,:,:,1] = torch.rand(*vec_grid[:,:,:,1].shape) * 2 - 1
    # vec_grid[:,:,:,2] = torch.rand(*vec_grid[:,:,:,2].shape) * 2 - 1
    # vec_grid /= vec_grid.square().sum(dim=-1).sqrt().unsqueeze(-1)
    
    # vec grid v1
    # grid_shape = (int(s[0] / res[0]) + 2, int(s[1] / res[1]) + 2, int(s[2] / res[2]) + 2)
    # angles = 2 * torch.pi * torch.rand(*grid_shape, device=device)
    # phi = torch.acos(2 * torch.rand(*grid_shape, device=device) - 1)
    
    # vec_grid = torch.stack([
    #     torch.sin(phi) * torch.cos(angles),
    #     torch.sin(phi) * torch.sin(angles),
    #     torch.cos(phi)
    # ], dim=-1)

    start = time.time()
    s000_loc = torch.floor(flat_grid).type(torch.int32)
    s001_loc = s000_loc  + torch.tensor([0, 0, 1], dtype=torch.int32)
    s010_loc = s000_loc  + torch.tensor([0, 1, 0], dtype=torch.int32)
    s011_loc = s000_loc  + torch.tensor([0, 1, 1], dtype=torch.int32)
    s100_loc = s000_loc  + torch.tensor([1, 0, 0], dtype=torch.int32)
    s101_loc = s000_loc  + torch.tensor([1, 0, 1], dtype=torch.int32)
    s110_loc = s000_loc  + torch.tensor([1, 1, 0], dtype=torch.int32)
    s111_loc = s000_loc  + torch.tensor([1, 1, 1], dtype=torch.int32)
    end = time.time()
    t4 = end - start

    print(flat_grid.shape, vec_grid.shape, s000_loc.shape)

    def grid_loc_dot(flat_grid, vec_grid, loc):
        return (-(flat_grid - loc) * vec_grid[loc[:,0], loc[:,1], loc[:,2]]).sum(dim=-1).clamp(-1, 1)
    
    s000 = grid_loc_dot(flat_grid, vec_grid, s000_loc)
    s001 = grid_loc_dot(flat_grid, vec_grid, s001_loc)
    s010 = grid_loc_dot(flat_grid, vec_grid, s010_loc)
    s011 = grid_loc_dot(flat_grid, vec_grid, s011_loc)
    s100 = grid_loc_dot(flat_grid, vec_grid, s100_loc)
    s101 = grid_loc_dot(flat_grid, vec_grid, s101_loc)
    s110 = grid_loc_dot(flat_grid, vec_grid, s110_loc)
    s111 = grid_loc_dot(flat_grid, vec_grid, s111_loc)


    smoothed_diff = Smooth(flat_grid - s000_loc)

    print(smoothed_diff.shape, s000.shape)
    x00 = s000 + (smoothed_diff[:,0]) * (s100 - s000)
    x01 = s001 + (smoothed_diff[:,0]) * (s101 - s001)
    x10 = s010 + (smoothed_diff[:,0]) * (s110 - s010)
    x11 = s011 + (smoothed_diff[:,0]) * (s111 - s011)
    
    y0 = x00 + (smoothed_diff[:,1]) * (x10 - x00)
    y1 = x01 + (smoothed_diff[:,1]) * (x11 - x01)
    
    z = y0 + (smoothed_diff[:,2]) * (y1 - y0)
    R = z.reshape(grid.shape[0:3])

    return R
