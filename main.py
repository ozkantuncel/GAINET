import torch
from src.data.dataset import DrugDataset
from src.data.dataloader import DrugDataLoader
from src.train import train
from src.test import test
from src.visualization.visualization import visualize, test_with_highlight_substructure
from src.models.mi_ddi import MI_DDI
from src.models.custom_sigmoid import SigmoidLoss
from src.utils.mol_features import TOTAL_ATOM_FEATS
from data.utils import train_tup, val_tup, test_tup, drug_id_to_smiles, drug_id_to_names
from config import CONFIG
from torch import optim



def select_mode():
    mode = input("Which mode would you like to run? (train/test/visualization): ").strip().lower()
    while mode not in ['train', 'test','visualization']:
        print("Invalid option. Please select either 'train', 'test' or visualization.")
        mode = input("Which mode would you like to run? (train/test/visualization): ").strip().lower()
    return mode


def main():

    train_data = DrugDataset(train_tup, ratio=CONFIG['data_size_ratio'], neg_ent=CONFIG['neg_samples'])
    val_data = DrugDataset(val_tup, ratio=CONFIG['data_size_ratio'], disjoint_split=False)
    test_data = DrugDataset(test_tup, disjoint_split=False)

    train_data_loader = DrugDataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_data_loader = DrugDataLoader(val_data, batch_size=CONFIG['batch_size'] *3)
    test_data_loader = DrugDataLoader(test_data, batch_size=CONFIG['batch_size'] *3)

    print(f"Training with {len(train_data_loader)} samples, validating with {len(val_data_loader)}, and testing with {len(test_data_loader)}")


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = MI_DDI(TOTAL_ATOM_FEATS,CONFIG['n_atom_hid'],CONFIG['kge_dim'],CONFIG['rel_total'], heads_out_feat_params=[64, 64, 64, 64], blocks_params=[2, 2, 2, 2])
    loss = SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay= CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    pkl_name = 'modelDDIMI.pkl'
    

    mode = select_mode()

    if mode == 'train':
        model.to(device=device)
        train(model, train_data_loader, val_data_loader, loss, optimizer, CONFIG['n_epochs'], device, scheduler=scheduler)
    elif mode == 'test':
        test_model= torch.load(pkl_name,map_location="cpu")
        model.load_state_dict(test_model)
        model.to(device=device)
        test(test_data_loader,model,device) 
    elif mode == 'visualization':
        test_model= torch.load(pkl_name,map_location="cpu")
        model.load_state_dict(test_model)
        model.to(device=device)
        visualize(test_data_loader, model, device)
        test_with_highlight_substructure(test_data_loader, model, device, drug_id_to_smiles, drug_id_to_names)


if __name__ == "__main__":
    main()