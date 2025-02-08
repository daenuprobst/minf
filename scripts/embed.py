import pickle
from pathlib import Path
import torch
import typer
from tqdm import tqdm
from minf.data import FieldDataset
from minf.models import Autoencoder, SirenNet
from minf.utils import plot_field
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_file: Path,
    output_file: Path,
    col_name: str = "smiles",
    resolution: int = 32,
    batch_size: int = 4,
    epochs: int = 1000,
):
    df = pd.read_csv(input_file)

    smiles = df["smiles"].tolist()

    fd = FieldDataset(smiles, resolution=resolution)
    dl = DataLoader(fd, batch_size=batch_size, shuffle=True)

    net = SirenNet(
        dim_in=4,
        dim_hidden=256,
        dim_out=1,
        num_layers=8,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    )

    autoencoder = Autoencoder(
        net,
        in_channels=10,
        output_shape=[resolution, resolution, resolution, 10, 1],
        latent_dim=256,
    )

    autoencoder.cuda()

    optim = torch.optim.Adam(lr=1e-4, params=autoencoder.parameters())

    min_loss = 9999999
    for epoch in (pbar := tqdm(range(500))):
        total_loss = 0
        for batch in dl:
            loss, _, _ = autoencoder(batch.cuda())
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(df)
        if avg_loss < min_loss:
            torch.save(autoencoder.state_dict(), str(output_file))
            min_loss = avg_loss
        pbar.set_description(f"Epoch {epoch} (loss: {avg_loss:2f})")


if __name__ == "__main__":
    app()
