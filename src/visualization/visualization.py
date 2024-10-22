import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.train import do_compute_with_attentions
import torch
from rdkit.Chem import Draw
import io
from rdkit.Chem import rdFMCS
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont


def visualize(test_data_loader, model, device, save_path='images', filename='attention_map'):
    model.eval()
    test_interaction_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            pos_score, neg_score, interaction_pred, true_label, attention = do_compute_with_attentions(batch, device, model)

            test_interaction_predictions.append(interaction_pred)
            test_true_labels.append(true_label)

    test_interaction_predictions = np.concatenate(test_interaction_predictions)
    test_true_labels = np.concatenate(test_true_labels)

    # Confusion Matrix
    conf_matrix = confusion_matrix(test_true_labels, np.round(test_interaction_predictions))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, f"{filename}_confusion_matrix.png")
    plt.savefig(full_path)
    plt.close()
    
def visualize_attentions(attention, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Sample Index")
    
    full_path = os.path.join(save_path, f"{filename}.png")
    plt.savefig(full_path)
    plt.close()
    

def test_with_highlight_substructure(test_data_loader, model, device, drug_id_to_smiles, drug_id_to_names, save_path='images/test_with_sub'):
    model.eval()
    test_probas_pred = []
    test_ground_truth = []
    test_attentions = []
    count = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            p_score, n_score, probas_pred, ground_truth, attentions  = do_compute_with_attentions(batch, device, model)

            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
            test_attentions.append(attentions)

        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)

        #TEST
        test_attentions = np.concatenate(test_attentions)

        #TEST
        max_attention = torch.max(attentions)
        normalized_attentions = attentions / max_attention

        count = 0
        for i, drug_pair in enumerate(test_data_loader.dataset):

            drug1_id, drug2_id, relation = drug_pair
            drug1_smiles = drug_id_to_smiles.get(drug1_id, "Unknown SMILES")
            drug2_smiles = drug_id_to_smiles.get(drug2_id, "Unknown SMILES")

            drug1_name = drug_id_to_names.get(drug1_id, drug1_id)
            drug2_name = drug_id_to_names.get(drug2_id, drug2_id)

            if count >= 20:
                break

            mol1 = Chem.MolFromSmiles(drug1_smiles)
            mol2 = Chem.MolFromSmiles(drug2_smiles)

            mcs = rdFMCS.FindMCS([mol1, mol2])
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

            substructure_atoms_mol1 = mol1.GetSubstructMatch(mcs_mol)
            substructure_atoms_mol2 = mol2.GetSubstructMatch(mcs_mol)

            substructure_bonds_mol1 = []
            substructure_bonds_mol2 = []

            for bond in mcs_mol.GetBonds():
                atom1_mol1 = substructure_atoms_mol1[bond.GetBeginAtomIdx()]
                atom2_mol1 = substructure_atoms_mol1[bond.GetEndAtomIdx()]
                bond_idx_mol1 = mol1.GetBondBetweenAtoms(atom1_mol1, atom2_mol1).GetIdx()
                substructure_bonds_mol1.append(bond_idx_mol1)

                atom1_mol2 = substructure_atoms_mol2[bond.GetBeginAtomIdx()]
                atom2_mol2 = substructure_atoms_mol2[bond.GetEndAtomIdx()]
                bond_idx_mol2 = mol2.GetBondBetweenAtoms(atom1_mol2, atom2_mol2).GetIdx()
                substructure_bonds_mol2.append(bond_idx_mol2)

            pil_img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2, subImgSize=(1920, 1080))


            draw = ImageDraw.Draw(pil_img)

            font_size = 25   
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default(size=font_size)

            img_width, img_height = pil_img.size
            text_y = img_height - font_size   
            
       
            draw.text(((1920, 1080)[0]//2, text_y ), drug1_name, font=font, fill=(0,0,0), anchor="mm")
            draw.text(((1920, 1080)[0] + (1920, 1080)[0]//2, text_y), drug2_name, font=font, fill=(0,0,0), anchor="mm")

            highlight_substructure(mol1, substructure_atoms_mol1, substructure_bonds_mol1, f'{drug1_name}-drug1.svg')
            highlight_substructure(mol2, substructure_atoms_mol2, substructure_bonds_mol2, f'{drug2_name}-drug2.svg')

            average_pred = (test_probas_pred[2*i] + test_probas_pred[2*i+1]) / 2
            real_pred = test_ground_truth[2*i] * test_ground_truth[2*i+1]

            plt.figure(figsize=(38.4, 28.8))  
            plt.imshow(pil_img)
            plt.text(0.5, 1, f"Interaction Prediction 1: {test_probas_pred[2*i]:.4f}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=22)
            plt.text(0.5, 0.95, f"Interaction Prediction 2: {test_probas_pred[2*i+1]:.4f}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=22)
            plt.text(0.5, 0.90, f"Average Interaction Prediction: {average_pred:.4f}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=22)
            plt.text(0.5, 0.85, f"Actual: {real_pred}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=22)
            plt.axis('off')

            output_path = os.path.join(save_path, f"{drug1_name}-{drug2_name} pair.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            count += 1



def highlight_substructure(mol, substructure_atoms, substructure_bonds, file_name, path='/Users/ozkantuncel/Desktop/mol_project/MI_DDI/images/higlight'):
    full_file_path = os.path.join(path, file_name)

    d = rdMolDraw2D.MolDraw2DCairo(700, 1000) 
    
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=substructure_atoms, highlightBonds=substructure_bonds)
    
    d.FinishDrawing()

    d.WriteDrawingText(full_file_path + '.png')