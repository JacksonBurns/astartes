with open("test\data\qm9_smiles.txt", "r") as file:
    lines = file.readlines()

qm9_smiles_short = [i.replace('\n', '') for i in lines[:100]]

qm9_smiles_full = [i.replace('\n', '') for i in lines]
