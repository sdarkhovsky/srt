import numpy as np
import imageio
import yaml
from torch.utils.data import Dataset

import os

from srt.utils.nerf import transform_points

class NMRDataset(Dataset):
    def __init__(self, path, mode, points_per_item=2048, max_len=None,
                 canonical_view=True, full_scale=False):
        """ Loads the NMR dataset as found at
        https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
        Hosted by Niemeyer et al. (https://github.com/autonomousvision/differentiable_volumetric_rendering)
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
        """
        self.path = path
        self.mode = mode
        self.points_per_item = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        self.full_scale = full_scale

        with open(os.path.join(path, 'metadata.yaml'), 'r') as f:
            metadata = yaml.load(f, Loader=yaml.CLoader)

        class_ids = [entry['id'] for entry in metadata.values()]

        self.scene_paths = []
        for class_id in class_ids:
            with open(os.path.join(path, class_id, f'softras_{mode}.lst')) as f:
                cur_scene_ids = f.readlines()
            cur_scene_ids = [scene_id.rstrip() for scene_id in cur_scene_ids if len(scene_id) > 1]
            cur_scene_paths = [os.path.join(class_id, scene_id) for scene_id in cur_scene_ids]
            self.scene_paths.extend(cur_scene_paths)
            
        self.num_scenes = len(self.scene_paths)
        print(f'NMR {mode} dataset loaded: {self.num_scenes} scenes.')

        self.render_kwargs = {
            'min_dist': 2.,
            'max_dist': 4.}

        # Rotation matrix making z=0 is the ground plane.
        # Ensures that the scenes are layed out in the same way as the other datasets,
        # which is convenient for visualization.
        self.rot_mat = np.array([[1, 0, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return self.num_scenes * 24

    def __getitem__(self, idx):
        scene_idx = idx % self.num_scenes
        view_idx = idx // self.num_scenes
        # view_idx: 8
        target_views = np.array(list(set(range(24)) - set([view_idx])))

        scene_path = os.path.join(self.path, self.scene_paths[scene_idx])
        images = [np.asarray(imageio.imread(
            os.path.join(scene_path, 'image', f'{i:04d}.png'))) for i in range(24)]
        images = np.stack(images, 0).astype(np.float32) / 255.
        # images.shape: (24, 64, 64, 3)
        input_image = np.transpose(images[view_idx], (2, 0, 1))

        cameras = np.load(os.path.join(scene_path, 'cameras.npz'))
        cameras = {k: v for k, v in cameras.items()}  # Load all matrices into memory

        for i in range(24): # Apply rotation matrix to rotate coordinate system
            cameras[f'world_mat_inv_{i}'] = self.rot_mat @ cameras[f'world_mat_inv_{i}'] 
            # The transpose here is not technically necessary, since the rotation matrix is symmetric
            # self.rot_mat is not symmetric os np.transpose is necessary (see details below)
            cameras[f'world_mat_{i}'] =  cameras[f'world_mat_{i}'] @ np.transpose(self.rot_mat)
        """
        cameras['world_mat_0']:
            array([[ 0.        ,  0.        , -1.        ,  0.        ],
                   [ 0.5       , -0.86602539,  0.        ,  0.        ],
                   [-0.86602539, -0.49999997,  0.        ,  2.73199987],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
        cameras['camera_mat_0']:
            array([[3.7320509, 0.       , 0.       , 0.       ],
                   [0.       , 3.7320509, 0.       , 0.       ],
                   [0.       , 0.       , 1.       , 0.       ],
                   [0.       , 0.       , 0.       , 1.       ]])
        A camera matrix in world coordinates: K⋅[I 0][R T]. It's 3x4. K 3x3 is the intrinsic camera matrix
                                                     [0 1]
        Xcam = [R T]⋅Xworld,  Xworld = [R T]⁻¹⋅Xcam       [R T]⁻¹ = [R⁻¹ R⁻¹T]
               [0 1]                   [0 1]              [0 1]     [0   1   ]
        The the camera origin in the world coordinates is [R⁻¹ R⁻¹T] [0] = [R⁻¹T], i.e. the last column
                                                          [0   1   ] [1]   [1   ]
        Xworld_rot = rot_mat⋅Xworld = rot_mat⋅[R T]⁻¹⋅Xcam      Xcam = [R T]⋅rot_mat⁻¹⋅Xworld_rot
                                              [0 1]                    [0 1]
        world_mat_{i} corresponds to [R T]     world_mat_inv_{i}   corresponds to [R T]⁻¹
                                     [0 1]                                        [0 1]
        camera_mat_{i} probably corresponds to the intrinsic camera matrix expanded to the size 4x4: [K 0]
                                                                                                     [0 1]
        Let K = [fx 0  0]      Ximg = K⋅[I 0]⋅Xcam = K [X Y Z]ᵀ = [fx⋅X fy⋅Y Z]ᵀ ≈ [fx⋅X/Z  fy⋅Y/Z  1]ᵀ
                [0  fy 0]
                [0  0  1]
        [Ximg] = [K 0]⋅[I 0]⋅Xcam =  [K 0]⋅Xcam          Xcam = [K 0]⁻¹ [Ximg]
        [1   ]   [0 1] [0 1]         [0 1]                      [0 1]   [1   ]
        """
        rays = []
        height = width = 64

        xmap = np.linspace(-1, 1, width)
        ymap = np.linspace(-1, 1, height)
        xmap, ymap = np.meshgrid(xmap, ymap)

        """
        Image coordinates of the pixels are (i,j), i < height, j < width
        Xcam = [i j 1]ᵀ
        """
        for i in range(24):
            cur_rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)
            cur_rays = transform_points(cur_rays,
                                        cameras[f'world_mat_inv_{i}'] @ cameras[f'camera_mat_inv_{i}'],
                                        translate=False)
            # cur_rays.shape (64, 64, 3)
            cur_rays = cur_rays[..., :3]
            cur_rays = cur_rays / np.linalg.norm(cur_rays, axis=-1, keepdims=True)
            rays.append(cur_rays)
            
        rays = np.stack(rays, axis=0).astype(np.float32)
        # the last column of world_mat_inv_X is the camera position in the world coordinates
        camera_pos = [cameras[f'world_mat_inv_{i}'][:3, -1] for i in range(24)]
        camera_pos = np.stack(camera_pos, axis=0).astype(np.float32)
        # camera_pos and rays are now in world coordinates.

        if self.canonical:  # Transform to canonical camera coordinates
            canonical_extrinsic = cameras[f'world_mat_{view_idx}'].astype(np.float32)
            camera_pos = transform_points(camera_pos, canonical_extrinsic)
            rays = transform_points(rays, canonical_extrinsic, translate=False)

        rays_flat = np.reshape(rays[target_views], (-1, 3))
        pixels_flat = np.reshape(images[target_views], (-1, 3))
        cpos_flat = np.tile(np.expand_dims(camera_pos[target_views], 1), (1, height * width, 1))
        cpos_flat = np.reshape(cpos_flat, (len(target_views) * height * width, 3))
        num_points = rays_flat.shape[0]

        if not self.full_scale:
            replace = num_points < self.points_per_item
            sampled_idxs = np.random.choice(np.arange(num_points),
                                            size=(self.points_per_item,),
                                            replace=replace)

            rays_sel = rays_flat[sampled_idxs]
            pixels_sel = pixels_flat[sampled_idxs]
            cpos_sel = cpos_flat[sampled_idxs]
        else:
            rays_sel = rays_flat
            pixels_sel = pixels_flat
            cpos_sel = cpos_flat

        result = {
            'input_images':      np.expand_dims(input_image, 0),              # [1, 3, h, w]
            'input_camera_pos':  np.expand_dims(camera_pos[view_idx], 0),     # [1, 3]
            'input_rays':        np.expand_dims(rays[view_idx], 0),           # [1, h, w, 3]
            'target_pixels':     pixels_sel,                                  # [p, 3]
            'target_camera_pos': cpos_sel,                                    # [p, 3]
            'target_rays':       rays_sel,                                    # [p, 3]
            'sceneid':           idx,                                         # int
        }

        if self.canonical:
            result['transform'] = canonical_extrinsic                         # [3, 4] (optional)

        return result

