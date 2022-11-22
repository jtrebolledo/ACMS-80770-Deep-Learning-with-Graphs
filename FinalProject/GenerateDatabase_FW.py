import maze
from maze.cif_download import download_cif

from ase.io import read
from ase.db import connect
import os

## Read the list of Zeolite Frameworks
with open('FW_List.txt') as f:
    lines = f.readlines()

## Generate Database with Zeolite FWs
con = connect('Database_FW.db', append=False)

## Iteration that read and write the zeolite FW in the generated database
for i in lines:
    name_FW = str(i[0:3])

    download_cif(name_FW)

    readstr = "data/" + name_FW + ".cif"

    # Try to read and write in the generated database
    try:
        # Read the atom file
        atoms = read(readstr)

        # Write in the generate database
        con.write(atoms, NameFramework = name_FW)
    except:
        pass

