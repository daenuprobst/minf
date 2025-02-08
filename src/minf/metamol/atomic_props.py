from typing import Dict
from elementy import PeriodicTable

periodic_table = PeriodicTable()


def symbol_to_atomic_number(symbol: str) -> int:
    return periodic_table.elements[symbol].atomic_number


def get_atomic_props() -> Dict[str, Dict[int, float]]:
    result = {}

    props = [
        "radius_vanDerWaals",
        "radius_covalent",
        "radius_metallic",
        "radius_USE",
        "electron_affinity",
        "wigner_seitz_electron_density",
        "electronegativity_pauling",
        "electronegativity_allen",
        "electronegativity_miedema",
        "electronegativity_mulliken",
        "mass",
    ]

    for prop in props:
        carbon_value = getattr(periodic_table.elements["C"], prop)

        if carbon_value is None:
            carbon_value = 1.0

        vals = {}
        for _, e in periodic_table.elements.items():
            attr = getattr(e, prop)
            if attr is None:
                vals[e.atomic_number] = 0.0
            else:
                vals[e.atomic_number] = attr / carbon_value

        result[prop] = vals

    return result
