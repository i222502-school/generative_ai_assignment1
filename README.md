# Generative AI — Assignment #1

Code for both questions. The written report is produced separately (Markdown handed off
for the student to convert to the required IEEE two-column PDF) and is not in this repo.

**Seed:** `42`, used for every data split, vocabulary build, and subset sample in both
questions (`q1_cnn_xray/src/config.py::SEED`, `q2_nmt_rnn/src/config.py::SEED`).

## Layout

```
q1_cnn_xray/   Question 1 -- chest X-ray CNN classification
q2_nmt_rnn/    Question 2 -- English-to-Urdu vanilla-RNN NMT
requirements.txt
```

Each question is a self-contained package (`src/`) plus a `colab_runner.ipynb` that
orchestrates it end to end on Google Colab, since neither question is realistically
trainable on CPU. Both notebooks mount Google Drive and cache all persistent state
(downloaded dataset, manifests, vocabularies, checkpoints, results) under
`/content/drive/MyDrive/generative_ai_assignment1`, so they resume rather than restart
across runtime disconnects.

## Local setup (development / smoke tests only — not full training)

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

CPU-only `torch`/`torchvision` is enough to run the modules against small inputs; real
training runs happen on Colab's GPU runtime via the two `colab_runner.ipynb` notebooks.

## Question 1 — grid search (Task 7)

`q1_cnn_xray/src/grid_search.py` runs a resumable 8-axis, 256-combination search over
the primary CNN's hyperparameters (batch size, learning rate, max epochs, dropout,
early-stopping patience, L1/L2 regularization, normalization scheme, augmentation
policy — see `full_grid()` for the exact value lists and the module docstring for why
it's shaped this way). It is driven from Section 6 of `q1_cnn_xray/colab_runner.ipynb`;
each completed combination is appended to `grid_search_results.csv` on Drive keyed by a
hash of its hyperparameters, so interrupting and re-running the cell skips finished
combinations rather than redoing them. `best_per_hyperparameter()` reduces the results to
the `hyperparameter · range · optimal value` table the report needs.

`q1_cnn_xray/src/data_volume_study.py` runs the companion 25/50/75/100% training-data
ablation from the same task, driven from Section 7 of the same notebook.

## Running the notebooks

1. Open `q1_cnn_xray/colab_runner.ipynb` (or `q2_nmt_rnn/colab_runner.ipynb`) in Colab.
2. Run the setup cells: mount Drive, clone this repo (prompts for a GitHub PAT), install
   `requirements.txt`.
3. Upload a Kaggle API token (`kaggle.json`, from kaggle.com/settings → API) when prompted,
   to download the dataset via `kagglehub`.
4. Run the remaining cells in order; each is labeled with the assignment task it covers.
