# POST-INFERENCE GUIDED TRANSFORMER FOR ANOMALY INTERVAL LOCALIZATION IN MULTIVARIATE TIME SERIES
This repository contains the implementation of a transformer-based framework for multiclass anomaly range detection in multivariate time series data, enhanced with post-inference strategies. Our method introduces domain-informed post-inference techniques, including majority voting and class transition masking, to stabilize multiclass anomaly interval predictions. The framework is evaluated on the  [Exathlon Benchmark Dataset](https://github.com/exathlonbenchmark/exathlon).

## Repository Structure
```bash
├── data/         # Preprocessed datasets across 3 folds 
├── diagrams/     # Design visuals
├── models/       # Saved models 
├── notebooks/    # Jupyter notebooks (e.g., pipeline.ipynb for reproducing results)
├── results/      # Generated results (plots, predictions)
├── src/          # Source code modules (data, model, loss, inference, evaluator, utils, HPO (hyperparameter optimization))
├── README.md     # Project overview (this file)
├── .gitignore    
├── requirements.txt # Python dependencies
```
## Getting Started
### Installation
You can install the dependencies using:
```bash
pip install -r requirements.txt
```
### Running the Pipeline
The main experiment notebook is provided in the notebooks/ directory.

Launch Jupyter Notebook and open:
```bash
notebooks/pipeline.ipynb
```
Run all cells sequentially to:
- Load and preprocess data
- Train models
- Apply post-inference techniques
- Save prediciton and evaluation resuls

