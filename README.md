# TCR-Embeddings

This repository aims to provide a framework for evaluating TCR Embeddings' expressivity with aim to understand the best way to map the functional landscape of T cell receptors.

Currently, support has been implemented for the following:

- Physico-chemical Embeddings: Atchley Factors [1], Kidera Factors [2], Amino Acid Properties [3], Random Embeddings (for control)
- LLM Embeddings: SCEPTR [4], TCR-BERT [5]

The code provides a flexible implementation with datasets, embedding models and hyperparameters to train models and subsequently using scripts to understand the effectiveness of the embeddings.

Should any user wishes to add support to more embedding models permanently, branching & submitting git merge requests are more than welcome, subject to satisfying Continuous-Integration requirements, detailed in subsequently sections.

>[!NOTE]
>If I have not responded to merge requests after some time, please feel free to [contact me](mailto:r.yuen.20@alumni.ucl.ac.uk).

## Installation Instructions

1. conda create --name tcr_embeddings python=3.12
2. conda activate tcr_embeddings
3. python -m pip install poetry
4. poetry install
5. python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## Usage Instructions (Training Models)
Default training hyperparameters have been placed under `tcr_embeddings/constants.json`.  You may change the hyperparameters of the training loop at your discretion.

>[!TIP]
>You may want to disable cuda for training fairly small models as GPU acceleration may not be beneficial for models with a small amount of parameters.

>[!TIP]
>Do **not** change `runtime_constants.py`

In your first-ever execution, run the following to generate the configuration file.
> `python -m tcr_embeddings.trainer --make`

This generates a `config.json` on your local directory.  In that file, you can change the more frequently changed parameters such as which fold within the k-Fold to run, which embedding method, reduction method, etc.

### Using your own Datasets
For any dataset, please clean the data such that the following is satisfied:
1. The data is in a `tsv` file format, the only difference to a csv is that a tsv uses tabs to seperate columns where csvs use commas.
2. The data has columns "TRAV", "TRBV", "CDR3A", "CDR3B".  You may put more columns in there at your discretion but this will slow down the execution through the data reading process.  Two sample files have been placed [here](data/sample.tsv) and [here](data/full.tsv).
3. Place your data in directory: `data/location`.  You may have multiple directories where the data shares the same label.

>[!TIP]
>To export your file to a csv, you can use ```df.to_csv("<filename>.tsv", sep="\t")```.

To generate consistent K-Fold cross validation sets (such that all training instances uses the same K-Fold), run the following after placing your data in the right location:

> `python -m data.create-kfold`

### Downloading Embeddings

- For TCR-BERT: 
   > `python -m tcr_embeddings.embed.download-tcrbert`
- For (re)creating Random Embeddings: 
   > `python -m tcr_embeddings.embed.create_random`

>[!NOTE]
>You cannot use the embedding method if you have not downloaded the embedding method.

### Training your Model

To train a model, run the following:
> `python -m tcr_embeddings.trainer --config <path/to/your/config>/config.json`

The training logs will be found within your specified output path in `config.json`, where you can generate the `config.json` file with `python -m tcr_embeddings.trainer --make`.

If your training has been paused / terminated in the middle and you wish to resume, run the following:

> `python -m tcr_embeddings.resume --dir path/to/your/resuming_instance`

Inside the output path, you will find the following:
1. Checkpoints for each epoch, alongside with the training loss csv, validation loss csv and testing loss csv
2. K-Folds used
3. Training Logs
4. Parquet Files of TCRs with non-zero weights, where the repertoire has been correctly classified within the test set.

## Usage Instructions (Analysis of Results)

We provide a few scripts to analyse your results, where they are placed under `analysis`.  The Jupyter notebooks provides already comprehensive description on what they are doing, however this is an overview:

* [interpretability-analysis.ipynb](analysis/interpretability-analysis.ipynb)

* [kfold-stats.ipynb](analysis/kfold-stats.ipynb)

* [training-stats-combined.ipynb](analysis/training-stats-combined.ipynb)

## CI/CD

To maintain good standards of code, the following CI/CD procedures must be complied and followed.  Unittests must be properly designed and covers edge cases.  CI procedures within the workflow must be fully passed prior to any origin/master branch merge requests.

> `black` > `isort` > `flake8` > `mypy` > `unittest` / `pytest`

Details for CI/CD can be found [here](.github/workflows/ci.yml)

## References

1. Atchley, W.R., Zhao, J., Fernandes, A.D., Druke, T.: Solving the protein sequence metric problem. Proceedings of the National Academy of Sciences 102(18), 6395–6400 (2005) https://doi.org/10.1073/pnas.0408677102
2. Kidera, A., Konishi, Y., Oka, M., Ooi, T., Scheraga, H.A.: Statistical analysis of the physical properties of the 20 naturally occurring amino acids. Journal of Protein Chemistry 4(1), 23–55 (1985) https://doi.org/10.1007/bf01025492
3.  Elhanati, Y., Sethna, Z., Marcou, Q., Callan, C.G., Mora, T., Walczak, A.M.: Inferring processes underlying b-cell repertoire diversity. Philosophical Transactions of the Royal Society B: Biological Sciences 370(1676), 20140243 (2015) https://doi.org/10.1098/rstb.2014.0243
4. Nagano, Y., Pyo, A., Milighetti, M., Henderson, J., Shawe-Taylor, J., Chain, B., Tiffeau-Mayer, A.: Contrastive learning of T cell receptor representations (2024). https://arxiv.org/abs/2406.06397
5. Wu, K., Yost, K.E., Daniel, B., Belk, J.A., Xia, Y., Egawa, T., Satpathy, A., Chang, H.Y., Zou, J.: Tcr-bert: learning the grammar of t-cell receptors for flexible antigen-xbinding analyses. TCR-Bert: Learning the grammar of T-cell receptors for flexible antigen-xbinding analyses (2021) https://doi.org/10.1101/2021.11.18.469186
