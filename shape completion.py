import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import os
import skimage
import trimesh
import numpy as np
import config_files
import yaml
import pybullet as pb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from results import runs_sdf
from model import vn_DeepSDF
import data.ShapeNetCoreV2 as ShapeNetCoreV2
import results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    training_settings_path = os.path.join(os.path.dirname(runs_sdf.__file__),  cfg['folder_sdf'], 'settings.yaml') 
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)

    return training_settings

def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis

def predict_sdf(latent, coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            latent_tile = torch.tile(latent, (coords.shape[0], 1))
            coords_latent = torch.hstack((latent_tile, coords))
            sdf_batch = model(coords_latent)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf

def extract_mesh(grad_size_axis, sdf):
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

    # Rescale vertices extracted with marching cubes (https://stackoverflow.com/questions/70834443/converting-indices-in-marching-cubes-to-original-x-y-z-space-visualizing-isosu)
    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces



def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def rotate_pointcloud(pointcloud_A, rpy_BA=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection.
    This calculates P_b, where P_b = R_b/a * P_a.
    R_b/a is rotation matrix of a wrt b frame.
    """
    # Rotate object
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    pointcloud_B = np.einsum('ij,kj->ki', rot_M, pointcloud_A)

    return pointcloud_B


def shapenet_rotate(mesh_original):
    '''In Shapenet, the front is the -Z axis with +Y still being the up axis. This function rotates the object to align with the canonical reference frame.
    Args:
        mesh_original: trimesh.Trimesh(), mesh from ShapeNet
    Returns:
        mesh: trimesh.Trimesh(), rotate mesh so that the front is the +X axis and +Y is the up axis.
    '''
    verts_original = np.array(mesh_original.vertices)

    rot_M = pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
    rot_M = np.array(rot_M).reshape(3, 3)
    verts = rotate_pointcloud(verts_original, [np.pi/2, 0, -np.pi/2])

    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_original.faces)

    return mesh



def generate_partial_pointcloud(cfg):
    """Load mesh and generate partial point cloud. The ratio of the visible bounding box is defined in the config file.
    Args:
        cfg: config file
    Return:
        samples: np.array, shape (N, 3), where N is the number of points in the partial point cloud.
        """
    # Load mesh
    obj_path = os.path.join(os.path.dirname(ShapeNetCoreV2.__file__), cfg['obj_ids'], 'models', 'model_normalized.obj')
    mesh_original = _as_mesh(trimesh.load(obj_path))

    # In Shapenet, the front is the -Z axis with +Y still being the up axis. 
    # Rotate objects to align with the canonical axis. 
    mesh = shapenet_rotate(mesh_original)

    # Sample on the object surface
    samples = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

    # Infer object bounding box and collect samples on the surface of the objects when the x-axis is lower than a certain threshold t.
    # This is to simulate a partial point cloud.
    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]

    v_min, v_max = mesh.bounds

    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]
    
    return samples


def main(cfg):
    model_settings = read_params(cfg)

    # Set directory and paths
    model_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'])

    inference_dir = os.path.join(model_dir, f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}")
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    output_mesh_path = os.path.join(inference_dir, 'output_mesh.obj')

    # Set tensorboard writer
    writer = SummaryWriter(log_dir=inference_dir, filename_suffix='inference_tensorboard')

    # Load the model
    weights = os.path.join(model_dir, 'weights.pt')

    model = vn_DeepSDF.SDFModel(
        num_layers=model_settings['num_layers'], 
        skip_connections=model_settings['latent_size'], 
        latent_size=model_settings['latent_size'], 
        inner_dim=model_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Define coordinates for mesh extraction
    coords, grad_size_axis = get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)

    # Generate partial point cloud
    pointcloud = generate_partial_pointcloud(cfg)

    # Save partial pointcloud
    pointcloud_path = os.path.join(inference_dir, 'partial_pointcloud.npy')
    np.save(pointcloud_path, pointcloud)

    # Generate torch tensors of zeros that has the same dimension as pointcloud
    pointcloud = torch.tensor(pointcloud, dtype=torch.float32).to(device)
    sdf_gt = torch.zeros_like(pointcloud[:, 0]).view(-1, 1).to(device)

    # Get the average optimised latent code
    results_path = os.path.join(model_dir, 'results.npy')
    results = np.load(results_path, allow_pickle=True).item()
    latent_code = results['best_latent_codes']
    # Get average latent code (across dimensions)
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
    latent_code.requires_grad = True
    
    # Infer latent code
    best_latent_code = model.infer_latent_code(cfg, pointcloud, sdf_gt, writer, latent_code)

    # Extract mesh obtained with the latent code optimised at inference
    sdf = predict_sdf(best_latent_code, coords_batches, model)
    vertices, faces = extract_mesh(grad_size_axis, sdf)
    output_mesh = _as_mesh(trimesh.Trimesh(vertices, faces))

    # Save mesh
    trimesh.exchange.export.export_mesh(output_mesh, output_mesh_path, file_type='obj')


if __name__ == '__main__':

    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'shape_completion.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)