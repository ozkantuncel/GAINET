import pandas as pd

df_ddi_train = pd.read_csv('data/ddi_training.csv')
df_ddi_test = pd.read_csv('data/ddi_test.csv')
df_ddi_val = pd.read_csv('data/ddi_validation.csv')
all_pos_tup = pd.read_csv('data/ddis.csv')
df_drugs_smiles = pd.read_csv('data/drug_smiles.csv')

drug_id_to_names_pd = pd.read_csv('data/file_drugs.csv')

drug_id_to_names = {drug_id: name for drug_id, name in drug_id_to_names_pd[['DrugBank ID', 'dg_name']].values}
drug_id_to_smiles = {drug_id: smiles for drug_id, smiles in df_drugs_smiles[['drug_id', 'smiles']].values}

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]

test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]


def totatl_prop():
    total = len(val_tup) + len(train_tup) + len(test_tup)
    print(len(train_tup) / total, len(test_tup)/total, len(val_tup)/total)


