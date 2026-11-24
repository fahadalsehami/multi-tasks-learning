# multi-tasks-learning: racial disparities in heart-disease mortality

A socio-demographic profile of **heart-disease mortality across U.S. states and counties**, and a **multi-task learning** prototype that predicts two outcomes from the same features at once.

## What the notebook does

1. **Builds one table from public data:** CDC 2015 mortality records (resident status, age, sex, race, marital status and place of death) joined by county FIPS code to American Community Survey comparative profiles (social, economic, housing and demographic), facility-level data, and CDC PLACES county estimates (2020 and 2021 releases) such as diagnosed diabetes, coronary heart disease, high cholesterol and lack of health insurance among adults.
2. **Cleans and engineers features:** normalizes census identifiers, profiles missing values, and merges the sources step by step into a final modelling dataset.
3. **Trains a multi-task neural network** (PyTorch, trained with fastai's one-cycle schedule) with a shared trunk and two heads: disease classification (accuracy) and age at death regression (RMSE). The two losses are balanced with learned, uncertainty-based weights, so neither task has to be hand-weighted.

> The multi-task section is a working prototype: it was developed on an artificial dataset with the same shape as the merged data, ready to be pointed at the real table.

## Run it

Open [`Notebook/heart_disease_mortality_mtl.ipynb`](Notebook/heart_disease_mortality_mtl.ipynb) in Jupyter or Google Colab. The notebook reads the source files from Google Drive; change the paths in the loading cells to where you keep the CDC and census extracts. The modelling cells use the fastai v1 API (`Learner` with `callback_fns`), so run them with fastai 1.0.x.

## Stack

Python, pandas, scikit-learn, PyTorch, fastai v1, Jupyter or Google Colab

## License

MIT, see [LICENSE](LICENSE).
