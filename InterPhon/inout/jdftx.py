import numpy as np
from echem.io_data.jdftx import Output
from echem.core.constants import Hartree2eV, Bohr2Angstrom


def read_output_lines(filepath: str, _) -> np.ndarray:
    _unit_convert = (1.602 * 10 ** (-19)) / 10 ** (-10)  # (eV/Angst) to (J/m)
    output = Output.from_file(filepath)
    forces = output.forces_hist[0] * Hartree2eV * _unit_convert / Bohr2Angstrom

    return forces
