# GAINET: Enhancing Drug-Drug Interaction Predictions Through Graph Neural Networks and Attention Mechanisms

- Das, Bihter, et al. **“GAINET: Enhancing Drug–Drug Interaction Predictions Through Graph Neural Networks and Attention Mechanisms.”** Chemometrics and Intelligent Laboratory Systems, vol. 259, 2025, p. 105337, https://doi.org/10.1016/j.chemolab.2025.105337.
  
- **Authors:** Bihter Das, Huseyin Alperen Dagdogen, Muhammed Onur Kaya, Ozkan Tuncel, Muhammed Samet Akgul, Resul Das

## Introduction

**GAINET** is a graph-based neural network model designed to predict drug-drug interactions (DDIs). In modern healthcare, accurately predicting DDIs is crucial, especially in cases of polypharmacy where multiple medications are used simultaneously. GAINET leverages molecular graph representations and attention mechanisms to focus on critical features of drug structures and their relationships, leading to improved accuracy in DDI predictions.

**Interpretability**: GAINET's attention mechanism allows for interpretable predictions by highlighting important molecular substructures. This feature makes it a valuable tool for safer drug development and treatment decisions. 

Flowchart for Drug-Drug Interaction Prediction with GAINET:

![flowchart-min](https://github.com/user-attachments/assets/6c5e89f9-b58e-4e2f-81f3-f09ba7b24938)

### Project Structure

```bash
GAINET/
├── data/
│   ├── ddi_test.csv
│   ├── ddi_training.csv
│   ├── ddi_validation.csv
│   ├── ddis.csv
│   ├── drug_smiles.csv
│   ├── file_drugs.csv
│   └── utils.py
├── images/
│   ├── higlight/
│   ├── test_with_sub/
│   └── attention_map_confusion_matrix.png
├── src/
│   ├── data/
│   │   ├── dataloader.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── custom_sigmoid.py
│   │   ├── layers.py
│   │   └── mi_ddi.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── mol_features.py
│   └── visualization/
│   │    └── visualization.py
│   └── test.py
│   └── train.py
├── .gitignore
├── config.py
├── main.py
├── README.md
└── requirements.txt
```

### Libraries Used in the GAINET Project

- **[RDKit](https://www.rdkit.org/):** A powerful tool for processing and analyzing molecular structures.
- **[PyTorch](https://pytorch.org/):** A popular library for building and training deep learning models.
- **[Torch Geometric](https://pytorch-geometric.com/):** A library built on top of PyTorch for applying deep learning models to graph data structures.
- **[Pandas](https://pandas.pydata.org/):** A powerful library for data manipulation and analysis.
- **[NumPy](https://numpy.org/):** A fundamental library for numerical computations.
- **[Matplotlib](https://matplotlib.org/):** A library used for creating plots and charts.
- **[Seaborn](https://seaborn.pydata.org/):** A library built on top of Matplotlib for statistical data visualization.
- **[Scikit-learn](https://scikit-learn.org/):** A popular library for data preprocessing and machine learning utilities. 


### Setup

Before running the application, ensure you have Python 3.9+ and install the required packages:

```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Run the application with the following command:

```bash
$ python main.py
```

### Running the Code
**Select Mode**
*  The program starts by prompting the user to select one of the modes: train, test, or visualization.

**Training**
* Training is selected, the model will be trained on the training dataset with real-time validation on the validation set.

**Testing**
* Testing is selected, the pre-trained model will be tested on the test dataset.

**Visualization**
* Visualization is selected, the model will produce visual reports and highlight important molecular structures.

### Key Functions

**Training**: The model learns drug-drug interaction patterns from the training dataset.

**Testing**  The model is evaluated on the test dataset to measure its performance on unseen data.

**Visualization**: The function visualizes attention mechanisms, confusion matrices, and important molecular substructures.

## Conclusion

In this study, we introduced GAINET, a graph-based neural network designed for predicting drug-drug interactions (DDIs). By combining graph neural networks with attention mechanisms, GAINET effectively captures complex molecular features and interaction dynamics, enabling precise and interpretable DDI predictions.

The model outperforms several existing methods in terms of accuracy, precision, recall, and other key metrics, based on evaluations using the DrugBank dataset. The attention mechanism enhances the model’s interpretability by highlighting relevant molecular substructures, making it highly useful for clinical decision-making.

Future work will focus on expanding GAINET’s capabilities through multi-task learning and application to larger, more diverse datasets, contributing to improved drug safety, personalized medicine, and more efficient drug development processes.
