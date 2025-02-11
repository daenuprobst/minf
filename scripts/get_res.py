from pathlib import Path
import typer
import torch
from minf.metamol import Metamol
from minf.models import Decoder, SirenNetUnbatched
from minf.models.utils import get_grid
from minf.utils import plot_field
import numpy as np

app = typer.Typer(pretty_exceptions_show_locals=False)


def calculate_psnr(original, compressed):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


@app.command()
def main(
    model_file: Path,
    out_file: Path,
    pdb_file: str,
    sdf_file: str,
    out_resolution: int = 64,
):
    resolution = 32
    rot = 2.9

    properties = [
        "wigner_seitz_electron_density",
    ]

    mmol, _ = Metamol.from_complex(
        pdb_file, sdf_file, size=out_resolution, properties=properties, b=0.25
    )

    mmol.calculate_field()
    vector_field = mmol.to_vector_field()[:, :, :, [0, 1, 9, 11, 12, 18]]

    net = SirenNetUnbatched(
        dim_in=4,
        dim_hidden=256,
        dim_out=1,
        num_layers=8,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    )

    decoder = Decoder(
        net,
        output_shape=[resolution, resolution, resolution, 6, 1],
        latent_dim=128,
        n_shapes=1,
    )

    decoder.cuda()

    decoder.load_state_dict(
        torch.load("../models/reconstruction_model_better.pt", weights_only=True)
    )
    decoder.cuda()

    superres = [out_resolution, out_resolution, out_resolution, 6]
    coords = get_grid(superres).cuda()

    reconstructed = decoder(None, latent_idx=0, coords=coords, output_shape=superres)
    reconstructed = reconstructed.cpu().detach().squeeze(-1)

    plot_field(reconstructed[:, :, :, 5], cam_rotate=rot, resolution=1024).save(
        "protein.png"
    )
    plot_field(reconstructed[:, :, :, 2], cam_rotate=rot, resolution=1024).save(
        "ligand.png"
    )

    print(calculate_psnr(vector_field.cpu().detach().numpy(), reconstructed.numpy()))


if __name__ == "__main__":
    app()
