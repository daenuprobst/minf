import torch
import yt
import numpy as np


def plot_field(
    field: torch.Tensor, resolution: int = 512, lens_type: str = "plane-parallel"
):
    arr = field.numpy()

    data = dict(density=(arr, "g/cm**3"))
    bbox = np.array([[0.0, 1.5], [0.0, 1.5], [0.0, 1.5]])
    ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", nprocs=64)
    sc = yt.create_scene(ds)

    cam = sc.add_camera(ds, lens_type=lens_type)
    cam.resolution = [resolution, resolution]
    return sc
