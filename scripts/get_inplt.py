import pickle
import torch
from tqdm import tqdm
from minf.data import FieldPDBDataset
from minf.models import Autoencoder, SirenNet, Decoder, SirenNetUnbatched
from minf.utils import plot_field
from torch.utils.data import DataLoader
import numpy as np
import yt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
    pdb_file: str,
):
    resolution = 32

    fd = FieldPDBDataset(pdb_file, resolution=resolution, single_property=9)
    dl_test = DataLoader(fd, batch_size=1, shuffle=False)

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
        # output_shape=[resolution, resolution, resolution, 24, 1],
        output_shape=[resolution, resolution, resolution, 1, 1],
        latent_dim=128,
        n_shapes=len(dl_test),
    )

    decoder.cuda(1)

    optim = torch.optim.Adam(lr=1e-4, params=decoder.parameters())

    min_loss = 9999999
    for epoch in (pbar := tqdm(range(50000))):
        for i, batch in enumerate(dl_test):
            total_loss = 0
            for field in batch:
                loss = decoder(field.cuda(), latent_idx=i)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dl_test)

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(decoder.state_dict(), model_file)

        pbar.set_description(f"Epoch {epoch} (loss: {avg_loss})")


if __name__ == "__main__":
    app()
