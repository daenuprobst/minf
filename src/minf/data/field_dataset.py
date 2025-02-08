from typing import List, Optional, Sequence
import torch
from torch.utils.data import Dataset

from minf.metamol import Metamol


class FieldDataset(Dataset):
    def __init__(
        self,
        smiles: Sequence[str],
        resolution: int = 16,
        b: float = 0.75,
        threshold: float = 0.5,
        properties: Optional[List] = None,
        radius_property: str = "radius_vanDerWaals",
        numconf: int = 1,
        mmff: bool = True,
        merge_conformers: bool = False,
        single_property: int = -1,
    ) -> None:
        self.smiles = smiles
        self.resolution = resolution
        self.b = b
        self.threshold = threshold
        self.properties = properties
        if self.properties is None:
            self.properties = [
                "wigner_seitz_electron_density",
            ]
        self.radius_property = radius_property
        self.numconf = numconf
        self.mmff = mmff
        self.merge_conformers = merge_conformers
        self.single_property = single_property

        self.cache = {}

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        smi = self.smiles[idx]

        mmol, _ = next(
            Metamol.from_smiles(
                smi,
                size=self.resolution,
                b=self.b,
                threshold=self.threshold,
                properties=self.properties,
                radius_property=self.radius_property,
                numconf=self.numconf,
                mmff=self.mmff,
                merge_conformers=self.merge_conformers,
            )
        )

        mmol.calculate_field(disable_progress_bar=True)
        field = None
        if self.single_property < 0:
            field = torch.nan_to_num(mmol.to_vector_field())
        else:
            field = torch.nan_to_num(
                mmol.to_vector_field()[:, :, :, self.single_property].unsqueeze(-1)
            )

        self.cache[idx] = field
        return field
