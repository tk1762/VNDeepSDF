import torch
import os
import trimesh
import skimage
import numpy as np
import config_files
import yaml
from results import runs_sdf
from model import vn_DeepSDF
import results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis

def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    
    training_settings_path = os.path.join(os.path.dirname(runs_sdf.__file__),  cfg['folder_sdf'], 'settings.yaml') 
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)
    print(training_settings)
    return training_settings

def reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis): 
    """
    Reconstruct the object from the latent code and save the mesh.
    Meshes are stored as .obj files under the same folder cerated during training, for example:
    - runs_sdf/<datetime>/meshes_training/mesh_0.obj
    """
    sdf = predict_sdf(latent_code, coords_batches, model)
    try:
        vertices, faces = extract_mesh(grad_size_axis, sdf)
    except:
        print('Mesh extraction failed')
        return
    
    # save mesh as obj
    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'meshes_training')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_idx}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')

def main(cfg):
    training_settings = read_params(cfg)

    # Load the mo.ptdel
    weights = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'weights.pt')

    model = vn_DeepSDF.SDFModel(
        num_layers=training_settings['num_layers'], 
        skip_connections=training_settings['latent_size'], 
        latent_size=training_settings['latent_size'], 
        inner_dim=training_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)
    
    # Load paths
    str2int_path = os.path.join(os.path.dirname(results.__file__), 'idx_str2int_dict.npy')
    results_dict_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'results.npy')
    
    # Load dictionaries
    str2int_dict = np.load(str2int_path, allow_pickle=True).item()
    results_dict = np.load(results_dict_path, allow_pickle=True).item()

    for obj_id_path in cfg['obj_ids']:
        # Get object index in the results dictionary
        obj_idx = str2int_dict[obj_id_path]  # index in collected latent vector
        # Get the latent code optimised during training
        latent_code = results_dict['best_latent_codes'][obj_idx]
        latent_code = torch.tensor(latent_code).to(device)

        reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis)


if __name__ == '__main__':

    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'reconstruct_from_latent.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)