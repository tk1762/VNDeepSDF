import numpy as np
import results 
import os
import point_cloud_utils as pcu
import data.ShapeNetCoreV2 as ShapeNetCoreV2
from glob import glob
from datetime import datetime
import config_files
import yaml
import pybullet as pb
import trimesh
import random
"""
For each object, sample points and store their distance to the nearest triangle.
Sampling follows the approach used in the DeepSDF paper.
"""

def SO3_rotate(vertices):

    # Generate random rotation
    angle = random.uniform(0, 2 * np.pi)  
    axis = np.array([0, 0, 1])  

    rotation_matrix = np.eye(3)  
    c = np.cos(angle)
    s = np.sin(angle)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                    [axis[2], 0, -axis[0]],
                                    [-axis[1], axis[0], 0]])
    rotation_matrix += s * cross_product_matrix + (1 - c) * np.outer(axis, axis)


    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    return(rotated_vertices)

def translation(vertices):

    # Generate random translation
    dx, dy, dz = np.random.rand(-0.5,0.5)

    translation_matrix = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ], dtype=float)

    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    translated_vertices = np.dot(homogeneous_vertices, translation_matrix.T)[:, :-1]
    return(translated_vertices)


def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   # repeat the latent code N times for stacking
    return np.hstack((latent_class_full, samples))

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
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    pointcloud_B = np.einsum('ij,kj->ki', rot_M, pointcloud_A)

    return pointcloud_B
def shapenet_rotate(mesh_original):
    
    verts_original = np.array(mesh_original.vertices)

    rot_M = pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
    rot_M = np.array(rot_M).reshape(3, 3)
    verts = rotate_pointcloud(verts_original, [np.pi/2, 0, -np.pi/2])

    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_original.faces)

    return mesh

def main(cfg):
  
    # Full paths to all .obj
    obj_paths = glob(os.path.join(os.path.dirname(ShapeNetCoreV2.__file__), '*', '*', 'models', '*.obj'))

    # File to store the samples and SDFs
    samples_dict = dict()        

    # Store conversion between object index (int) and its folder name (str)
    idx_str2int_dict = dict()
    idx_int2str_dict = dict()

    for obj_idx, obj_path in enumerate(obj_paths):

        # Object unique index. Str to int by byte encoding
        obj_idx_str = '/'.join(obj_path.split(os.sep)[-4:-2]) # e.g. '02958343/1a2b3c4d5e6f7g8h9i0j'
        idx_str2int_dict[obj_idx_str] = obj_idx
        idx_int2str_dict[obj_idx] = obj_idx_str

        # Dictionary to store the samples and SDFs
        samples_dict[obj_idx] = dict()


        #Converting models to standard models
        try:
            verts, faces = pcu.load_mesh_vf(obj_path)

            # Convert to watertight mesh
            mesh_original = _as_mesh(trimesh.load(obj_path))
            
            if not mesh_original.is_watertight:
                verts, faces = pcu.make_mesh_watertight(mesh_original.vertices, mesh_original.faces, 50000)
                mesh_original = trimesh.Trimesh(vertices=verts, faces=faces)
        except Exception as e:
            print(e)
            continue
        mesh = shapenet_rotate(mesh_original)
        verts = np.array(mesh.vertices)

        # random SO(3) rotation and translation
        verts = SO3_rotate(translation(verts))

        # Generate random points in the predefined volume that surrounds all the shapes.
        # NOTE: ShapeNet shapes are normalized within [-1, 1]^3
        p_vol = np.random.rand(cfg['num_samples_in_volume'], 3) * 2 - 1

        # Sample within the object's bounding box. This ensures a higher ratio between points inside and outside the surface.
        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=[v_min[0], v_min[1], v_min[2]], high=[v_max[0], v_max[1], v_max[2]], size=(cfg['num_samples_in_bbox'], 3))

        # Sample points on the surface as face ids and barycentric coordinates
        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])

        # Compute 3D coordinates and normals of surface samples
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        p_total = np.vstack((p_vol, p_bbox, p_surf))

        
        # Comput the SDF of the random points
        sdf, _, _  = pcu.signed_distance_to_mesh(p_total, verts, faces)

        samples_dict[obj_idx]['sdf'] = sdf
        # The samples are p_total, while the latent class is [obj_idx]
        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32))
    
    np.save(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{cfg["dataset"]}.npy'), samples_dict)

    np.save(os.path.join(os.path.dirname(results.__file__), f'idx_str2int_dict.npy'), idx_str2int_dict)
    np.save(os.path.join(os.path.dirname(results.__file__), f'idx_int2str_dict.npy'), idx_int2str_dict)

    return samples_dict, idx_str2int_dict, idx_int2str_dict




if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_sdf.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)