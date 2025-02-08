from typing import Dict, Optional
import numpy as np


class Metaatom:
    """
    A class representing a atom with properties such as atomic number, coordinates,
    radius, electronegativity, and additional custom properties.

    Attributes:
        atomic_number (int): The atomic number of the atom.
        v (np.ndarray): A 3D vector representing the coordinates (x, y, z) of the atom.
        props (Dict[str, float]): A dictionary of additional properties associated with the atom.
        complex (int): A property representing the complexity of the atom.
        r (float): The radius of the atom.
        en (float): The electronegativity of the atom.
    """

    def __init__(
        self,
        atomic_number: int,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        r: float = 0.0,
        en: float = 0.0,
        props: Optional[Dict[str, float]] = None,
        complex: int = 0,
    ) -> None:
        """
        Initializes a Metaatom instance.

        Args:
            atomic_number (int): The atomic number of the atom.
            x (float, optional): The x-coordinate of the atom. Defaults to 0.0.
            y (float, optional): The y-coordinate of the atom. Defaults to 0.0.
            z (float, optional): The z-coordinate of the atom. Defaults to 0.0.
            r (float, optional): The radius of the atom. Defaults to 0.0.
            en (float, optional): The electronegativity of the atom. Defaults to 0.0.
            props (Optional[Dict[str, float]], optional): Additional properties of the atom. Defaults to None.
            complex (int, optional): A property representing the complexity of the atom. Defaults to 0.
        """
        if props is None:
            props = {}

        self.__atomic_number = atomic_number
        self.v = np.array([x, y, z], dtype=float)
        self.props = props
        self.complex = complex
        self.r = r
        self.en = en

    @property
    def x(self) -> float:
        """float: The x-coordinate of the atom."""
        return self.v[0]

    @x.setter
    def x(self, value: float) -> None:
        self.v[0] = value

    @property
    def y(self) -> float:
        """float: The y-coordinate of the atom."""
        return self.v[1]

    @y.setter
    def y(self, value: float) -> None:
        self.v[1] = value

    @property
    def z(self) -> float:
        """float: The z-coordinate of the atom."""
        return self.v[2]

    @z.setter
    def z(self, value: float) -> None:
        self.v[2] = value

    @property
    def atomic_number(self) -> int:
        """int: The atomic number of the atom."""
        return self.__atomic_number

    @atomic_number.setter
    def atomic_number(self, value: int) -> None:
        self.__atomic_number = value

    def set_coords(self, coords: np.ndarray) -> None:
        """
        Sets the coordinates of the atom using a numpy array.

        Args:
            coords (np.ndarray): A numpy array containing the x, y, and z coordinates.
        """
        if coords.shape != (3,):
            raise ValueError("Coordinates must be a 3-element array.")
        self.x, self.y, self.z = coords
