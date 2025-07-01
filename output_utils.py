# output_utils.py

import os
import numpy as np

def print_cell_dimensions(crystal, output_dir=None):
    cell = crystal.get_cell()
    max_x = np.linalg.norm(cell[0])
    max_y = np.linalg.norm(cell[1])
    max_z = np.linalg.norm(cell[2])

    msg = (
        f"Crystal cell dimensions (Å):\n"
        f"  x: {max_x:.3f}\n"
        f"  y: {max_y:.3f}\n"
        f"  z: {max_z:.3f}\n"
        f"Suggested cavity centers:\n"
        f"  center:        {[max_x/2, max_y/2, max_z/2]}\n"
        f"  bottom-right:  {[max_x*0.9, max_y*0.1, max_z/2]}\n"
        f"  top-left:      {[max_x*0.1, max_y*0.9, max_z/2]}\n"
    )
    print(msg)
    if output_dir:
        with open(f"{output_dir}/cell_dimensions.txt", "w") as f:
            f.write(msg)


def print_crystal_info(crystal, label="", output_dir=None):
    cell = crystal.get_cell()
    # Fix ptp call here
    size = np.ptp(crystal.get_positions(), axis=0)
    info = (f"{label} crystal info:\n"
            f"  Number of atoms: {len(crystal)}\n"
            f"  Cell vectors:\n{cell}\n"
            f"  Atomic positions range (Å):\n  x: {size[0]:.3f}, y: {size[1]:.3f}, z: {size[2]:.3f}\n")
    print(info)
    if output_dir:
        with open(os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_info.txt"), "w") as f:
            f.write(info)


