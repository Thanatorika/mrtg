import json
import numpy as np
from sklearn import datasets
import h5py
from torch_topological.datasets.spheres import create_sphere_dataset

def make_mammoth(filepath, noise_level=0.02):
    """
    Load mammoth data from a JSON file and preprocess it.
    
    Args:
        filepath (str): Path to the JSON file containing mammoth data.
        noise_level (float): Level of noise to add to the data.
        
    Returns:
        tuple: A tuple containing the mammoth points and colors.
    """
    with open(filepath, 'r') as file:
        mammoth_data = json.load(file)

    mammoth_points = np.array(mammoth_data)
    mammoth_points = mammoth_points[:, [0,2,1]]

    max_val, min_val = mammoth_points.max(), mammoth_points.min()
    mammoth_points = (mammoth_points - min_val) / (max_val - min_val)

    noise = np.random.rand(10000, 3) * noise_level
    noisy_mammoth = mammoth_points + noise

    mammoth_color = mammoth_points[:, 2]

    return noisy_mammoth, mammoth_color

def make_swiss_roll(n_samples=2000, noise_level=0.02):
    """
    Load Swiss roll data from a JSON file and preprocess it.
    
    Args:
        n_samples (int): Number of samples to generate.
        noise_level (float): Level of noise to add to the data.
        
    Returns:
        tuple: A tuple containing the Swiss roll points and colors.
    """
    sr_points, sr_color = datasets.make_swiss_roll(n_samples=n_samples, hole=True, random_state=0)

    min, max = sr_points.min(), sr_points.max()
    sr_points = (sr_points - min) / (max - min)

    noise = np.random.randn(n_samples, 3) * noise_level
    noisy_sr = sr_points + noise

    return noisy_sr, sr_color

def make_partnet(filepath):
    """
    Load PartNet data from a HDF5 file and preprocess it.
    
    Args:
        filepath (str): Path to the HDF5 file containing PartNet data.
        
    Returns:
        tuple: A tuple containing the PartNet shapes, points, and segmentations.
    """

    dataset = h5py.File(filepath, 'r')

    shapes = np.array(dataset['shapes']).astype(str)
    pts = np.array(dataset['points'])
    segmentation = np.array(dataset['segmentations'])

    return shapes, pts, segmentation

def make_spheres(n_samples=200, n_spheres=9, dimension=100, radius=5, noise_level=0.02):
    """
    Generate synthetic sphere data.
    
    Args:
        n_samples (int): Number of samples in each sphere to generate.
        n_spheres (int): Number of spheres to generate.
        dimension (int): Intrinsic dimension of the spheres.
        radius (int): Radius of the spheres.
        noise_level (float): Level of noise to add to the data.
        
    Returns:
        tuple: A tuple containing the sphere points and colors.
    """
    spheres_pt, spheres_color = create_sphere_dataset(n_samples=n_samples, n_spheres=n_spheres, d=dimension, r=radius, seed=None)

    return spheres_pt, spheres_color
