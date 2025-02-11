from pathlib import Path
from typing import List, Optional, Sequence
import torch
from torch.utils.data import Dataset

from minf.metamol import Metamol


class FieldPDBDataset(Dataset):
    def __init__(
        self,
        pdb_file: Path | str,
        resolution: int = 16,
        b: float = 0.75,
        threshold: float = 0.5,
        properties: Optional[List] = None,
        radius_property: str = "radius_vanDerWaals",
        single_property: int = -1,
        subset: Optional[List] = None,
    ) -> None:
        self.path = pdb_file
        self.resolution = resolution
        self.b = b
        self.threshold = threshold
        self.properties = properties
        if self.properties is None:
            self.properties = [
                "wigner_seitz_electron_density",
            ]
        self.radius_property = radius_property
        self.single_property = single_property

        self.conformers = [
            mmol
            for mmol, _ in Metamol.from_pdb_conformers(
                self.path,
                size=self.resolution,
                b=self.b,
                threshold=self.threshold,
                properties=self.properties,
                radius_property=self.radius_property,
            )
        ]

        self.subset = subset

        if self.subset is None:
            self.subset = list(range(len(self.conformers)))

        self.cache = {}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        idx = self.subset[idx]
        if idx in self.cache:
            return self.cache[idx]
        conformer = self.conformers[idx]

        conformer.calculate_field(disable_progress_bar=True)

        field = None
        if self.single_property < 0:
            field = torch.nan_to_num(conformer.to_vector_field())
        else:
            field = torch.nan_to_num(
                conformer.to_vector_field()[:, :, :, self.single_property].unsqueeze(-1)
            )

        self.cache[idx] = field
        return field
