import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_peptides(n):
    print("Generating peptides...")

    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    peptides = []

    def build_peptide(current_peptide):
        if len(current_peptide) == n:
            peptides.append(current_peptide)
            return
        for amino_acid in amino_acids:
            build_peptide(current_peptide + amino_acid)

    build_peptide('')

    print(f"Generated {len(peptides)} peptides.")
    
    return peptides

def predictPeptide():
    peptides = generate_peptides(3)
    morgan_peptides = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromFASTA(peptide), 4, 4096) for peptide in peptides]
    morgan_peptides_df = pd.DataFrame(np.array(morgan_peptides, int))

    # fp_dfと同じ変数を持つようにする
    morgan_peptides_df = morgan_peptides_df.loc[:, morgan_peptides_df.columns.isin(fp_df.columns)]

    # モデルによる予測
    morgan_peptides_df = morgan_peptides_df.iloc[:, feat_selector.support_]

    result = rf.predict(morgan_peptides_df)
    print(result)