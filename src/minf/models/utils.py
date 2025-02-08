import torch


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


def get_grid(output_shape, min_val=-1, max_val=1):
    tensors = [torch.linspace(min_val, max_val, steps=i) for i in output_shape]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(output_shape))
    return mgrid
