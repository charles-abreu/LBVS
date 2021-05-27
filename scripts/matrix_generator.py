# IMPORTS
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from glob import glob
import os, sys

def get_matrix(smiles_file, reduce=False):
  with open(smiles_file) as in_file:

    matrix = []
    names = []
    for line in in_file:
      if '\t' in line:
        smiles = line.split('\t')[0].strip()
        name = line.split('\t')[1].strip()
      else:
        smiles = line.split(' ')[0].strip()
        name = line.split(' ')[1].strip()

      molecule = Chem.MolFromSmiles(smiles)
      if molecule is None:
        continue

      # Topological Fingerprints 2048 bits
      fp1 = Chem.RDKFingerprint(molecule)
      # MACCS Keys 167 bits
      fp2 = MACCSkeys.GenMACCSKeys(molecule)
      # Concatena fingerprints
      fp_final = list(fp1.ToBitString() + fp2.ToBitString())
      # tranforma em uma lista de inteiros
      matrix.append([int(x) for x in fp_final])
      names.append(name)

    df = pd.DataFrame(matrix, index=names)
    df.index.name = "name"
    return df

# data_dir: path containing data (each target is a a directory with smiles)
# st_type: T = test, V = validation
def run_LIT(data_dir, set_type):
    dir_list = glob(data_dir + os.sep + "*")

    for pdb in dir_list:
        if set_type == "T" and os.path.exists(pdb + os.sep +'train.csv'): continue
        if set_type == "V" and os.path.exists(pdb + os.sep +'test.csv'): continue

        pdb_id = os.path.basename(pdb)
        print(pdb_id)
        # smiles files
        active_smiles = pdb + os.sep + pdb_id + '_active_' + set_type + '.smiles'
        inactive_smiles = pdb + os.sep + pdb_id + '_inactive_'+ set_type +'.smiles'
        # Gerando matriz para ligantes
        active_df = get_matrix(active_smiles)
        active_df['class'] = 1
        # Gerando matriz para decoys
        inactive_df = get_matrix(inactive_smiles)
        inactive_df['class'] = 0
        # Concatenando matrizes
        matrix = pd.concat([active_df, inactive_df])
        # Salvando matriz no Drive
        if set_type == "V":
            matrix.to_csv(pdb + os.sep +'test.csv')
        elif set_type == "T":
            matrix.to_csv(pdb + os.sep +'train.csv')

def run_5HT(data_dir):
    # smiles files
    active_smiles = data_dir + os.sep + 'ligands.smiles'
    inactive_smiles = data_dir + os.sep + 'decoys.smiles'
    # Gerando matriz para ligantes
    active_df = get_matrix(active_smiles)
    active_df['class'] = 1
    # Gerando matriz para decoys
    inactive_df = get_matrix(inactive_smiles)
    inactive_df['class'] = 0
    # Concatenando matrizes
    matrix = pd.concat([active_df, inactive_df])
    # Salvando matriz no Drive
    matrix.to_csv(data_dir + os.sep +'data.csv')

if __name__ == '__main__':
    active_smiles = sys.argv[1]
    inactive_smiles = sys.argv[2]
    # Gerando matriz para ligantes
    active_df = get_matrix(active_smiles)
    active_df['class'] = 1
    # Gerando matriz para decoys
    inactive_df = get_matrix(inactive_smiles)
    inactive_df['class'] = 0
    # Concatenando matrizes
    matrix = pd.concat([active_df, inactive_df])
    # Salvando matriz no Drive
    matrix.to_csv(os.path.dirname(active_smiles) + os.sep +'data.csv')
