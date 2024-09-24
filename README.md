
# Your Repo Has Been Successfully Created

### Install Conda Environment
```
conda env create -f environment.yaml
conda activate cp_tx_py38
```

### Data Preparation
Firstly,  create an empty folder called 'data'
Then, run the notebook 'generate_dummy_data.ipynb' in folder 'src'. This will generate some dummy Cell Painting and Transcriptomics data for you.
If you don't have data and just want to see the code run on dummy data, do nothing at this step. Else, if you have your own Cell Painting/Transcriptomics data, format your data and rename it according to the dummy files generated in the 'data' folder.

### Pretraining and Embedding Generation
Finally, in the 'src' folder, run the run.sh file to train the Contrastive Learning and Bimodal Autoencoder model, and generate embeddings in the data/embeddings folder.
```
bash run.sh
```