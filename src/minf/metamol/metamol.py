from typing import Union, List, Dict, Generator, Iterable, Tuple, Optional
from pathlib import Path
from PIL import Image as im

import torch

import numpy as np

from matplotlib import cm
from rdkit.Chem import (
    AllChem,
    SDMolSupplier,
    MolFromPDBFile,
    MolFromSmiles,
    MolToMolBlock,
    MolFromMolBlock,
    Mol,
    CombineMols,
    AddHs,
    rdDistGeom,
    rdMolAlign,
    rdDepictor,
)

from scipy.ndimage import gaussian_filter

from tqdm import tqdm

from minf.metamol.metaatom import Metaatom
from .atomic_props import get_atomic_props


@torch.jit.script
def _generate_base_field(n: int, center: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Generate a 3D tensor field based on the given parameters.

    This function generates a 3D tensor field where each point in the field is calculated
    based on its distance from a specified center point, scaled by a given factor.

    Args:
        n (int): The size of the tensor field along each dimension. This defines the resolution of the field.
        center (torch.Tensor): A tensor of shape (3,) representing the center point of the field.
        scale (float): A scaling factor used to adjust the influence of the distance from the center.

    Returns:
        torch.Tensor: A 3D tensor of shape (n, n, n) representing the generated field.
    """
    s = torch.linspace(0, 1.0, n)
    sx = ((s - center[0]) * (1 / scale)) ** 2
    sy = ((s - center[1]) * (1 / scale)) ** 2
    sz = ((s - center[2]) * (1 / scale)) ** 2

    return (
        torch.unsqueeze(torch.unsqueeze(sx, -1), -1)
        + torch.unsqueeze(torch.unsqueeze(sy, 0), -1)
        + torch.unsqueeze(torch.unsqueeze(sz, 0), 0)
    )


@torch.jit.script
def get_atom_field(
    n: int,
    center: torch.Tensor,
    r: float,
    beta: float = 0.5,
    beta_shape: Optional[float] = None,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate atom fields based on given parameters.

    Args:
        n (int): The size of the field.
        center (torch.Tensor): The center coordinates for the field generation.
        r (float): The radius parameter.
        beta (float, optional): The beta parameter for the first field. Default is 0.5.
        beta_shape (float, optional): The beta parameter for the second field. If None, the second field is not generated. Default is None.
        scale (float, optional): The scaling factor for the field. Default is 1.0.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - field_a (torch.Tensor): The generated field with the first beta parameter.
            - field_b (torch.Tensor, optional): The generated field with the second beta parameter, if beta_shape is provided; otherwise, None.
    """

    field = _generate_base_field(n, center, scale)

    field_a = torch.exp(-beta * ((field / r**2) - 1))
    field_a /= torch.max(field_a)

    if beta_shape is None:
        return field_a, None

    field_b = torch.exp(-beta_shape * ((field / r**2) - 1))
    field_b /= torch.max(field_b)

    return field_a, field_b


@torch.jit.script
def get_atom_fields(
    n: int,
    center: torch.Tensor,
    radii: List[float],
    beta: float = 0.5,
    scale: float = 1.0,
) -> List[torch.Tensor]:
    """
    Generates atom fields for a given set of radii centered at a specific point.

    Args:
        n (int): The size of the field.
        center (torch.Tensor): The coordinates of the center point.
        radii (List[float]): A list of radii for which the fields are to be generated.
        beta (float, optional): A scaling factor for the exponential function. Default is 0.5.
        scale (float, optional): A scaling factor for the base field. Default is 1.0.

    Returns:
        List[torch.Tensor]: A list of tensor fields for each radius.
    """
    field = _generate_base_field(n, center, scale)
    fields = []

    for r_i in radii:
        field_i = torch.exp(-beta * ((field / r_i**2) - 1))
        field_i /= torch.max(field_i)
        fields.append(field_i)

    return fields


class Metamol:
    atomic_props: Dict[str, Dict[int, float]] = get_atomic_props()

    def __init__(
        self,
        size: int = 128,
        threshold: float = 0.5,
        b: float = 1.0,
        b_shape: Optional[float] = None,
        atoms: Optional[List[Metaatom]] = None,
        radius_property: str = "radius_vanDerWaals",
        properties: Optional[List] = None,
        scale: float = 1.0,
        is_complex: bool = False,
    ) -> None:
        """
        Initialize the class with the following parameters.

        Args:
            size (int, optional): The size parameter. Defaults to 128.
            threshold (float, optional): The threshold value. Defaults to 0.5.
            b (float, optional): The 'b' parameter. Defaults to 1.0.
            b_shape (float, optional): The shape parameter for 'b'. Defaults to None.
            atoms (List[Metaatom], optional): A list of atoms. Defaults to None.
            radius_property (str, optional): The property name for radius. Defaults to "radius_vanDerWaals".
            properties (List, optional): Additional properties. Defaults to None.
            scale (float, optional): The scaling factor. Defaults to 1.0.
            is_complex (bool, optional): Boolean flag indicating if the object is complex. Defaults to False.
        """
        if atoms is None:
            atoms = []

        self.atoms = atoms
        self.size = size
        self.threshold = threshold
        self.b = b
        self.b_shape = b_shape
        self.radius_property = radius_property
        self.properties = properties if properties is not None else []
        self.scale = scale
        self.is_complex = is_complex
        self.field_cache = {}
        self.field_a = None
        self.field_b = None
        self.property_fields_a = {}
        self.property_fields_b = {}
        self.property_fields_cache = {}
        self.element_fields_a = {}
        self.element_fields_b = {}
        self.mesh = None
        self.original_vert_coords = None

        self.n_atoms_in_field = 0

    def add(self, matms: Union[Metaatom, Iterable[Metaatom]]):
        """
        Adds one or more Metaatom objects to the atoms list.

        Args:
            matms (Union[Metaatom, Iterable[Metaatom]]): A single Metaatom object or an iterable of
                                                         Metaatom objects to be added to the atoms list.

        Returns:
            None
        """
        if isinstance(matms, Iterable):
            for matm in matms:
                self.atoms.append(matm)
        else:
            self.atoms.append(matms)

    def to_vector_field(self, elements: Optional[List[int]] = None):
        """
        Converts atomic and property fields into a vector field.

        This function takes atomic and property fields, combines them, and returns a vector field
        representation. If no elements are provided, a default list of elements is used. The resulting
        vector field includes channels for the specified elements, an additional channel for elements
        not in the list, and channels for properties. If the field is complex, additional channels
        are included for the two parts of the complex (e.g. a ligand and a protein).

        Parameters:
            elements (List[int], optional): A list of atomic numbers to include in the vector field.
                                            If None or empty, a default list is used.

        Returns:
            torch.Tensor: A tensor representing the combined vector field.
        """
        if elements is None or len(elements) < 1:
            elements = [1, 6, 7, 8, 9, 15, 16, 17]

        # Add one for "others", elements not in list
        n_channels = len(elements) + 1 + len(self.properties)

        if self.is_complex:
            n_channels *= 2

        vector_field = torch.from_numpy(
            np.zeros((*self.field_a.shape, n_channels), dtype=np.float32)
        )

        for atomic_number, atomic_field in self.element_fields_a.items():
            if atomic_number in elements:
                vector_field[:, :, :, elements.index(atomic_number)] = atomic_field
            else:
                vector_field[:, :, :, int(n_channels / 2) - 1] += atomic_field

        for property_name, property_field in self.property_fields_a.items():
            vector_field[
                :, :, :, len(elements) + 1 + self.properties.index(property_name)
            ] += property_field

        if self.is_complex:
            for atomic_number, atomic_field in self.element_fields_b.items():
                if atomic_number in elements:
                    vector_field[
                        :, :, :, int(n_channels / 2) + elements.index(atomic_number)
                    ] = atomic_field
                else:
                    vector_field[:, :, :, 2 * len(elements) - 1] += atomic_field

            for property_name, property_field in self.property_fields_b.items():
                vector_field[
                    :,
                    :,
                    :,
                    2 * len(elements) + 2 + self.properties.index(property_name),
                ] = property_field

        return torch.nan_to_num(vector_field)

    def to_image(self, path, bins=0, smooth=0.0):
        """
        Converts a 3D field tensor to a series of 2D images and saves them to the specified path.

        Args:
            path (str): The directory path where the images will be saved.
            bins (int, optional): The number of bins to discretize the field values. Default is 0 (no discretization).
            smooth (float, optional): The sigma value for Gaussian smoothing. Default is 0.0 (no smoothing).

        Returns:
            None

        The method performs the following steps:
            1. Converts the `path` to a string if it is not already.
            2. If `bins` is greater than 0, it discretizes the `field` values into the specified number of bins.
            3. If `smooth` is greater than 0.0, it applies Gaussian smoothing to the `field`.
            4. Iterates through the slices of the `field` tensor along the second dimension and converts each slice to a 2D image.
            5. The images are saved in the specified `path` with filenames in the format "path.i.png", where `i` is the slice index.

        """
        path = str(path)
        field = self.field.clone()
        if bins > 0:
            bins = np.linspace(0.0, 1.0, num=bins)
            centers = (bins[1:] + bins[:-1]) / 2
            field = torch.tensor(bins[np.digitize(field.numpy(), centers)])

        if smooth > 0.0:
            field = gaussian_filter(field, sigma=smooth)

        for i in range(len(field)):
            section = (cm.turbo(torch.clip(field[:, i, :], 0, 1)) * 255).astype(
                np.uint8
            )
            data = im.fromarray(section)
            data.save(path + "." + str(i) + ".png")

    def process_atom_field(self, atom: Metaatom) -> None:
        """
        Process the field of a given atom and update various field caches and properties.

        This function updates the fields for the atom based on its properties, adding the
        calculated fields to the appropriate caches and aggregating the fields as needed.

        Args:
            atom (Metaatom): An instance of the Metaatom class representing the atom to process.

        Steps:
            1. Retrieve the standard field (e.g., van der Waals radius) from the cache.
            2. If the field is not in the cache, calculate it and add it to the cache.
            3. If the atom does not belong to a complex or belongs to the primary complex, update field_a;
               otherwise, update field_b.
            4. Update the per-atom fields for element_fields_a or element_fields_b based on the complex status.
            5. Calculate and update property fields if they exist.
            6. Increment the count of atoms in the field(s).
        """
        # Get the standard field (e.g. van der waals radius)
        field, shape_field = self.get_cached_field(
            self.field_cache, atom.v, atom.atomic_number
        )

        if field is None:
            v = np.array([0.5, 0.5, 0.5], dtype=float)
            field, shape_field = get_atom_field(
                self.size,
                torch.FloatTensor(v),
                atom.r,
                self.b,
                self.b_shape,
                self.scale,
            )

            pad_dim = int(self.size / 2)

            field = torch.nn.functional.pad(
                field,
                (pad_dim, pad_dim, pad_dim, pad_dim, pad_dim, pad_dim),
                mode="constant",
            ).to(torch.float16)

            if shape_field is not None:
                shape_field = torch.nn.functional.pad(
                    shape_field,
                    (pad_dim, pad_dim, pad_dim, pad_dim, pad_dim, pad_dim),
                    mode="constant",
                ).to(torch.float16)

            self.field_cache[atom.atomic_number] = (field, shape_field)

            field, shape_field = self.get_cached_field(
                self.field_cache, atom.v, atom.atomic_number
            )

        if not self.is_complex or self.is_complex and atom.complex == 0:
            if shape_field is None:
                self.field_a += field
            else:
                self.field_a += shape_field
        else:
            if shape_field is None:
                self.field_b += field
            else:
                self.field_b += shape_field

        # Set the per-atom fields
        if not self.is_complex or self.is_complex and atom.complex == 0:
            if atom.atomic_number not in self.element_fields_a:
                self.element_fields_a[atom.atomic_number] = field
            else:
                self.element_fields_a[atom.atomic_number] += field
        else:
            if atom.atomic_number not in self.element_fields_b:
                self.element_fields_b[atom.atomic_number] = field
            else:
                self.element_fields_b[atom.atomic_number] += field

        # Calculate property fields if they exist
        if len(self.property_fields_a) > 1:
            fields = self.get_cached_property_field(
                self.property_fields_cache, atom.v, atom.atomic_number
            )

            if fields is None:
                v = np.array([0.5, 0.5, 0.5], dtype=float)
                fields = get_atom_fields(
                    self.size,
                    torch.FloatTensor(v),
                    [atom.props[prop] for prop in self.properties],
                    self.b,
                    self.scale,
                )

                pad_dim = int(self.size / 2)

                for i in range(len(fields)):
                    fields[i] = torch.nn.functional.pad(
                        fields[i],
                        (pad_dim, pad_dim, pad_dim, pad_dim, pad_dim, pad_dim),
                        mode="constant",
                    ).to(torch.float16)

                self.property_fields_cache[atom.atomic_number] = fields

                fields = self.get_cached_property_field(
                    self.property_fields_cache, atom.v, atom.atomic_number
                )

            if not self.is_complex or self.is_complex and atom.complex == 0:
                for i, property_name in enumerate(self.properties):
                    self.property_fields_a[property_name] += fields[i]
            else:
                for i, property_name in enumerate(self.properties):
                    self.property_fields_b[property_name] += fields[i]

        self.n_atoms_in_field += 1

    # TODO: Refactor to use tensors
    def get_cached_field(
        self, cache: Dict, center: np.ndarray, atomic_number: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves a cached field for a given atomic number and center.


        This function checks if the atomic number exists in the cache. If it does,
        it retrieves the cached property fields, adjusts them based on the given center,
        and returns the adjusted fields.

        Args:
            cache (Dict): A dictionary where keys are atomic numbers and values are tuples
                          of cached field arrays and their shapes.
            center (np.ndarray): The center coordinates to be used for slicing the cached field.
            atomic_number (int): The atomic number used as a key to retrieve the cached field from the cache.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the sliced cached field and its shape.
            Returns (None, None) if the atomic number is not in the cache.

        """
        if atomic_number not in cache:
            return None, None

        cf, cf_shape = cache[atomic_number]

        n = self.size

        c_cached = np.array([cf.shape[0] / 2] * 3, dtype=int)
        c = (center * n).astype(int)
        f = c_cached - c
        t = c_cached + n - c

        cf = cf[f[0] : t[0], f[1] : t[1], f[2] : t[2]].clone()

        if cf_shape is not None:
            c_cached = np.array([cf_shape.shape[0] / 2] * 3, dtype=int)
            c = (center * n).astype(int)
            f = c_cached - c
            t = c_cached + n - c

            cf_shape = cf_shape[f[0] : t[0], f[1] : t[1], f[2] : t[2]].clone()

        return cf, cf_shape

    # TODO: Refactor to use tensors
    def get_cached_property_field(
        self, cache: Dict, center: np.ndarray, atomic_number: int
    ) -> Optional[List[np.ndarray]]:
        """
        Retrieves a cached property field for a given atomic number and center.

        This function checks if the atomic number exists in the cache. If it does,
        it retrieves the cached property fields, adjusts them based on the given center,
        and returns the adjusted fields.

        Args:
            cache (Dict): A dictionary where keys are atomic numbers and values are lists of cached property fields.
            center (np.ndarray): A numpy array representing the center coordinates.
            atomic_number (int): The atomic number for which the cached property field is to be retrieved.

        Returns:
            Optional[List[np.ndarray]]: A list of adjusted property fields if the atomic number exists in the cache,
            otherwise None.
        """
        if atomic_number not in cache:
            return None

        cached_cfs = cache[atomic_number]
        out_cfs = []
        n = self.size

        for cf in cached_cfs:
            c_cached = np.array([cf.shape[0] / 2] * 3, dtype=int)
            c = (center * n).astype(int)
            f = c_cached - c
            t = c_cached + n - c

            out_cfs.append(cf[f[0] : t[0], f[1] : t[1], f[2] : t[2]].clone())

        return out_cfs

    def calculate_field(
        self, ignore_hydrogens: bool = False, disable_progress_bar: bool = True
    ) -> None:
        """
        Calculate the field for the molecular system. This involves initializing atomic tensor fields
        and processing atoms to populate these fields. Optionally ignores hydrogen atoms and can disable
        the progress bar for the operation.

        Args:
            ignore_hydrogens (bool): If True, hydrogen atoms will be ignored in the field calculation.
                                     The default is False.
            disable_progress_bar (bool): If True, the progress bar will be disabled during the field calculation.
                                         The default is True.

        Returns:
            None
        """
        self.field_a = torch.zeros((self.size, self.size, self.size))

        if self.is_complex:
            self.field_b = torch.zeros((self.size, self.size, self.size))

        if len(self.properties) > 0:
            self.property_fields_a = {
                property_name: torch.zeros((self.size, self.size, self.size))
                for property_name in self.properties
            }

            if self.is_complex:
                self.property_fields_b = {
                    property_name: torch.zeros((self.size, self.size, self.size))
                    for property_name in self.properties
                }

        self.element_fields_a = {}
        self.element_fields_b = {}
        self.n_atoms_in_field = 0

        if disable_progress_bar:
            for atom in self.atoms:
                if not (ignore_hydrogens and atom.atomic_number == 1):
                    self.process_atom_field(atom)
        else:
            for atom in tqdm(self.atoms, leave=False):
                if not (ignore_hydrogens and atom.atomic_number == 1):
                    self.process_atom_field(atom)

        # In case there are only hydrogen atoms in the molecule
        if len(self.atoms) > 0 and self.n_atoms_in_field == 0:
            self.calculate_field(
                ignore_hydrogens=ignore_hydrogens,
                disable_progress_bar=disable_progress_bar,
            )

        # Norm the fields to [0, 1]
        self.field_a = self.field_a.div(torch.max(self.field_a))

        if self.is_complex:
            self.field_b = self.field_b.div(torch.max(self.field_b))

        for k in self.element_fields_a.keys():
            self.element_fields_a[k] = self.element_fields_a[k].div(
                torch.max(self.element_fields_a[k])
            )

        for k in self.element_fields_b.keys():
            self.element_fields_b[k] = self.element_fields_b[k].div(
                torch.max(self.element_fields_b[k])
            )

        for k in self.property_fields_a.keys():
            self.property_fields_a[k] = self.property_fields_a[k].div(
                torch.max(self.property_fields_a[k])
            )

        if self.is_complex:
            for k in self.property_fields_b.keys():
                self.property_fields_b[k] = self.property_fields_b[k].div(
                    torch.max(self.property_fields_b[k])
                )

    def sample_field(self, x: np.ndarray) -> float:
        """
        Calculate the sample field value at a given point.

        Args:
            x (np.ndarray): A numpy array representing the coordinates of the point at which
                            the sample field value is to be calculated. Must be (x,y,z).

        Returns:
            float: The calculated sample field value at the given point `x`.
        """
        val = 0.0
        for atom in self.atoms:
            r = (
                Metamol.atomic_props[self.radius_property][atom.atomic_number]
                * self.scale
            )
            c = atom.v
            v = np.linalg.norm(x - c)
            val += np.exp(-v / (2 * r**2))

        return val

    @staticmethod
    def from_rdkit_mol(
        mol,
        size: int = 128,
        threshold: float = 1.0,
        b: float = 1.0,
        b_shape: float = None,
        radius_property: str = "radius_vanDerWaals",
        properties: Optional[List] = None,
    ) -> "Metamol":
        """
        Create a Metamol object from an RDKit molecule.

        This function converts an RDKit molecule into a Metamol object. It extracts
        atomic coordinates, atomic numbers, and other properties, scales the coordinates,
        and initializes Metaatom instances for each atom in the molecule.

        Args:
            mol: An RDKit molecule object.
            size (int): The size parameter for the Metamol object. Default is 128.
            threshold (float): The threshold parameter for the Metamol object. Default is 1.0.
            b (float): The b parameter for the Metamol object. Default is 1.0.
            b_shape (float): The b_shape parameter for the Metamol object. Default is None.
            radius_property (str): The property name for atomic radius. Default is "radius_vanDerWaals".
            properties (List, optional): A list of additional properties to extract for each atom. Default is None.

        Returns:
            Metamol: A Metamol object initialized with the provided RDKit molecule.
        """
        if properties is None:
            properties = []

        coords, atomic_numbers, complex = [], [], []

        conformer = mol.GetConformer(0)

        for i, atom in enumerate(mol.GetAtoms()):
            positions = conformer.GetAtomPosition(i)
            atomic_numbers.append(atom.GetAtomicNum())

            if atom.HasProp("complex"):
                complex.append(int(atom.GetProp("complex")))

            coords.append([positions.x, positions.y, positions.z])

        coords = np.array(coords)
        atomic_numbers = np.array(atomic_numbers)
        is_complex = True

        if not complex:
            complex = [0] * len(atomic_numbers)
            is_complex = False

        matms = []

        for c, atomic_number, comp in zip(coords, atomic_numbers, complex):
            radius = Metamol.atomic_props[radius_property][atomic_number]
            matms.append(
                Metaatom(
                    atomic_number,
                    c[0],
                    c[1],
                    c[2],
                    radius,
                    complex=comp,
                    props={
                        prop: Metamol.atomic_props[prop][atomic_number]
                        for prop in properties
                    },
                )
            )

        max_r = max([matm.r for matm in matms])

        centroid = np.mean(coords, axis=0)
        coords -= centroid

        max_distance = np.max(np.sqrt(np.sum(abs(coords) ** 2, axis=-1)))
        max_distance += np.sqrt(((max_r**2 * np.log(threshold)) / -b) + 1)

        scale = 1.0 / (2 * max_distance)
        # scale = (size - 1) / (2 * (max_distance + max_r))

        # hacky fix for single [H]
        if scale == 0:
            scale = 1.0

        coords *= scale
        coords += 0.5

        mmol = Metamol(
            threshold=threshold,
            b=b,
            b_shape=b_shape,
            size=size,
            radius_property=radius_property,
            properties=properties,
            scale=scale,
            is_complex=is_complex,
        )

        for i in range(len(matms)):
            matms[i].set_coords(coords[i])

        mmol.add(matms)

        return mmol

    @staticmethod
    def from_multi_rdkit_mols(
        mols: Iterable,
        size: int = 128,
        threshold: float = 1.0,
        b: float = 1.0,
        b_shape: float = None,
        radius_property: str = "radius_vanDerWaals",
        properties: Optional[List] = None,
    ) -> "Metamol":
        """
        Create a Metamol object from multiple RDKit molecules.

        This function converts multiple RDKit molecules into a single Metamol object. It extracts
        atomic coordinates, atomic numbers, and other properties from each molecule, scales the coordinates,
        and initializes Metaatom instances for each atom. As multiple molecules are supplied, this happens for
        each of the molecules. All atoms are combined within a single Metamol molecule.

        Args:
            mols (Iterable): An iterable of RDKit molecule objects.
            size (int): The size parameter for the Metamol object. Default is 128.
            threshold (float): The threshold parameter for the Metamol object. Default is 1.0.
            b (float): The b parameter for the Metamol object. Default is 1.0.
            b_shape (float): The b_shape parameter for the Metamol object. Default is None.
            radius_property (str): The property name for atomic radius. Default is "radius_vanDerWaals".
            properties (List, optional): A list of additional properties to extract for each atom. Default is None.

        Returns:
            Metamol: A Metamol object initialized with the provided RDKit molecules.
        """
        coords = []
        atomic_numbers = []
        complex = []

        if properties is None:
            properties = []

        for mol in mols:
            conformer = mol.GetConformer()

            for i, atom in enumerate(mol.GetAtoms()):
                positions = conformer.GetAtomPosition(i)
                atomic_numbers.append(atom.GetAtomicNum())
                if atom.HasProp("complex"):
                    complex.append(int(atom.GetProp("complex")))

                coords.append([positions.x, positions.y, positions.z])

        coords = np.array(coords)
        atomic_numbers = np.array(atomic_numbers)
        matms = []

        if not complex:
            complex = [0] * len(atomic_numbers)

        for c, atomic_number, comp in zip(coords, atomic_numbers, complex):
            radius = Metamol.atomic_props[radius_property][atomic_number]
            matms.append(
                Metaatom(
                    atomic_number,
                    c[0],
                    c[1],
                    c[2],
                    radius,
                    complex=comp,
                    props={
                        prop: Metamol.atomic_props[prop][atomic_number]
                        for prop in properties
                    },
                )
            )

        max_r = max([matm.r for matm in matms])

        max_distance = np.max(np.sqrt(np.sum(abs(coords) ** 2, axis=-1)))
        max_distance += np.sqrt(((max_r**2 * np.log(threshold)) / -b) + 1)
        scale = 1.0 / (2.0 * max_distance)
        coords *= scale

        # hacky fix for single [H]
        if scale == 0:
            scale = 1.0

        coords += 0.5

        mmol = Metamol(
            threshold=threshold,
            b=b,
            b_shape=b_shape,
            size=size,
            radius_property=radius_property,
            scale=scale,
            is_complex=bool(complex),
        )

        for i in range(len(matms)):
            matms[i].set_coords(coords[i])

        mmol.add(matms)

        return mmol

    @staticmethod
    def from_sdf(
        path: Union[str, Path],
        size: int = 128,
        threshold: float = 1.0,
        b: float = 1.0,
        b_shape: float = None,
        radius_property: str = "radius_vanDerWaals",
        removeHs: bool = True,
        numconf: int = 1,
        mmff: bool = True,
        merge_conformers: bool = False,
    ) -> Generator[Tuple["Metamol", Mol], None, None]:
        """
        Create Metamol objects from an SDF file.

        This function reads an SDF file and generates Metamol objects from the molecules
        contained in the file. It can handle multiple conformers and optionally merge them
        into a single Metamol object.

        Args:
            path (Union[str, Path]): Path to the SDF file.
            size (int): The size parameter for the Metamol object. Default is 128.
            threshold (float): The threshold parameter for the Metamol object. Default is 1.0.
            b (float): The b parameter for the Metamol object. Default is 1.0.
            b_shape (float): The b_shape parameter for the Metamol object. Default is None.
            radius_property (str): The property name for atomic radius. Default is "radius_vanDerWaals".
            removeHs (bool): Whether to remove hydrogen atoms. Default is True.
            numconf (int): Number of conformers to generate. Default is 1.
            mmff (bool): Whether to use MMFF for conformer generation. Default is True.
            merge_conformers (bool): Whether to merge multiple conformers into a single Metamol object. Default is False.

        Yields:
            Generator[Tuple[Metamol, Mol], None, None]: A generator yielding tuples of Metamol objects
                                                        and their corresponding RDKit molecules.
        """
        suppl = SDMolSupplier(str(path), removeHs=removeHs)

        for mol in suppl:
            if numconf < 2:
                try:
                    yield (
                        Metamol.from_rdkit_mol(
                            mol, size, threshold, b, b_shape, radius_property
                        ),
                        mol,
                    )
                except Exception:
                    return None, mol
            else:
                if merge_conformers:
                    conformers = []
                    for conformer in Metamol.mol_to_conformers(mol, numconf, mmff):
                        conformers.append(conformer)

                    yield (
                        Metamol.from_multi_rdkit_mols(
                            conformers, size, threshold, b, b_shape, radius_property
                        ),
                        conformers,
                    )
                else:
                    for conformer in Metamol.mol_to_conformers(mol, numconf, mmff):
                        yield (
                            Metamol.from_rdkit_mol(
                                conformer, size, threshold, b, b_shape, radius_property
                            ),
                            conformer,
                        )

    @staticmethod
    def from_pdb(
        path: Union[str, Path],
        size: int = 128,
        threshold: float = 1.0,
        b: float = 1.0,
        b_shape: Optional[float] = None,
        radius_property: str = "radius_vanDerWaals",
        removeHs: bool = True,
    ) -> Tuple["Metamol", Mol]:
        """
        Create a Metamol object and an RDKit Mol object from a PDB file.

        Args:
            path (Union[str, Path]): The path to the PDB file.
            size (int, optional): The size parameter for the Metamol object. Defaults to 128.
            threshold (float, optional): The threshold parameter for the Metamol object. Defaults to 1.0.
            b (float, optional): The b parameter for the Metamol object. Defaults to 1.0.
            b_shape (float, optional): The b_shape parameter for the Metamol object. Defaults to None.
            radius_property (str, optional): The radius property to use for the Metamol object. Defaults to "radius_vanDerWaals".
            properties (List, optional): Additional properties for the Metamol object. Defaults to None.
            removeHs (bool, optional): Whether to remove hydrogens. Defaults to True.

        Returns:
            Tuple["Metamol", Mol]: A tuple containing the Metamol object and the combined RDKit Mol object.
        """

        mol = MolFromPDBFile(str(path), removeHs=removeHs)
        return (
            Metamol.from_rdkit_mol(mol, size, threshold, b, b_shape, radius_property),
            mol,
        )

    @staticmethod
    def from_complex(
        pdb_path: Union[str, Path, Mol],
        sdf_path: Union[str, Path, Mol],
        size: int = 128,
        threshold: float = 0.5,
        b: float = 1.0,
        b_shape: Optional[float] = None,
        radius_property: str = "radius_vanDerWaals",
        properties: Optional[List] = None,
        removeHs: bool = False,
        sanitize: bool = False,
    ) -> Tuple["Metamol", Mol]:
        """
        Creates a Metamol object from a complex of PDB and SDF files.

        Args:
            pdb_path (Union[str, Path, Mol]): Path to the PDB file or an RDKit Mol object.
            sdf_path (Union[str, Path, Mol]): Path to the SDF file or an RDKit Mol object.
            size (int, optional): Size parameter for Metamol. Defaults to 128.
            threshold (float, optional): Threshold parameter for Metamol. Defaults to 0.5.
            b (float, optional): Parameter b for Metamol. Defaults to 1.0.
            b_shape (float, optional): Shape parameter b for Metamol. Defaults to None.
            radius_property (str, optional): Radius property for atoms. Defaults to "radius_vanDerWaals".
            removeHs (bool, optional): Whether to remove hydrogens. Defaults to False.
            sanitize (bool, optional): Whether to sanitize the molecule. Defaults to False.

        Returns:
            Tuple["Metamol", Mol]: A tuple containing the Metamol object and the combined RDKit Mol object.
        """
        if properties is None:
            properties = []

        pdb_mol = pdb_path
        sdf_mol = sdf_path

        if isinstance(pdb_mol, (str, Path)):
            pdb_mol = MolFromPDBFile(str(pdb_mol), removeHs=removeHs, sanitize=sanitize)

        if isinstance(sdf_mol, (str, Path)):
            sdf_mol = next(
                SDMolSupplier(str(sdf_mol), removeHs=removeHs, sanitize=sanitize)
            )

        for atom in pdb_mol.GetAtoms():
            atom.SetProp("complex", "1")

        for atom in sdf_mol.GetAtoms():
            atom.SetProp("complex", "0")

        combined_mol = CombineMols(pdb_mol, sdf_mol)

        return (
            Metamol.from_rdkit_mol(
                combined_mol, size, threshold, b, b_shape, radius_property, properties
            ),
            combined_mol,
        )

    # Adapted from https://iwatobipen.wordpress.com/2021/01/31/
    # generate-conformers-script-with-rdkit-rdkit-chemoinformatics/
    @staticmethod
    def mol_to_conformers(mol, numconf: int = 10, mmff: bool = True) -> List["Metamol"]:
        """
        Generate multiple conformers for a given molecule.

        Args:
            mol (rdkit.Chem.Mol): The input molecule.
            numconf (int): The number of conformers to generate. Default is 10.
            mmff (bool): Whether to perform MMFF optimization on the conformers. Default is True.

        Returns:
            List[Metamol]: A list of conformers in Metamol format.

        The function performs the following steps:
            1. Adds hydrogen atoms to the molecule.
            2. Attempts to generate multiple conformers using the EmbedMultipleConfs method.
            3. If no conformers are generated, it uses a different set of parameters to attempt again.
            4. If still no conformers are generated, it computes 2D coordinates and sets them as 3D.
            5. Optionally performs MMFF optimization on the conformers.
            6. Aligns the conformers.
            7. Converts the conformers to Metamol format and returns them.
        """
        mol = AddHs(mol)

        conformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=numconf, randomSeed=42, clearConfs=True
        )

        skip_mmff = False

        if len(conformers) == 0:
            params = rdDistGeom.ETKDGv3()
            params.maxAttempts = 999
            params.useRandomCoords = True
            conformers = rdDistGeom.EmbedMultipleConfs(mol, numconf, params)
            skip_mmff = True

        if len(conformers) == 0:
            # If all else fails, just get 2d coords
            rdDepictor.Compute2DCoords(mol)
            mol.GetConformer().Set3D(True)
            skip_mmff = True

        if mmff and not skip_mmff:
            for i in range(len(conformers)):
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=i)
                except Exception:
                    continue

        rdMolAlign.AlignMolConformers(mol)

        return [
            MolFromMolBlock(MolToMolBlock(mol, confId=conf), removeHs=False)
            for conf in conformers
        ]

    @staticmethod
    def from_smiles(
        smiles: str,
        size: int = 128,
        threshold: float = 0.5,
        b: float = 1.0,
        b_shape: Optional[float] = None,
        radius_property: str = "radius_vanDerWaals",
        properties: Optional[List] = None,
        numconf: int = 10,
        mmff: bool = True,
        merge_conformers: bool = False,
    ) -> Generator[Tuple["Metamol", Union[Mol, List[Mol]]], None, None]:
        """
        Generate Metamol objects from a SMILES string.

        This method generates molecular conformers from a given SMILES string and processes them
        into Metamol objects. It allows control over various parameters related to the conformer generation
        and Metamol creation process.

        Args:
            smiles (str): The SMILES representation of the molecule.
            size (int, optional): The size parameter for the Metamol object. Defaults to 128.
            threshold (float, optional): The threshold parameter for the Metamol object. Defaults to 0.5.
            b (float, optional): The b parameter for the Metamol object. Defaults to 1.0.
            b_shape (Optional[float], optional): The b_shape parameter for the Metamol object. Defaults to None.
            radius_property (str, optional): The radius property to use for the Metamol object. Defaults to "radius_vanDerWaals".
            properties (Optional[List], optional): Additional properties for the Metamol object. Defaults to None.
            numconf (int, optional): Number of conformers to generate. Defaults to 10.
            mmff (bool, optional): Whether to use MMFF (Merck Molecular Force Field) for conformer generation. Defaults to True.
            merge_conformers (bool, optional): Whether to merge conformers into a single Metamol object. Defaults to False.

        Yields:
            Generator[Tuple["Metamol", Union[Mol, List[Mol]]], None, None]: A generator yielding tuples of Metamol objects and their corresponding conformers.
        """

        mol = MolFromSmiles(smiles)

        conformers = list(Metamol.mol_to_conformers(mol, numconf, mmff))

        if merge_conformers:
            metamol = Metamol.from_multi_rdkit_mols(
                conformers, size, threshold, b, b_shape, radius_property, properties
            )
            yield metamol, conformers
        else:
            for conformer in conformers:
                metamol = Metamol.from_rdkit_mol(
                    conformer,
                    size,
                    threshold,
                    b,
                    b_shape,
                    radius_property,
                    properties,
                )
                yield metamol, conformer
