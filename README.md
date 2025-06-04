# LLM Model Routing

This repository contains the code to replicate the experiments and generate the essential plots for the paper [*Large Language Model Routing with Benchmark Datasets*](https://openreview.net/forum?id=Zb0ajZ7vAt) by Shnitzer et al. The experiment was run on Python 3.9.5 with the requirements found in `requirements.txt`

## Main Expeirment

To run the main experiment use the command:

`python run_experiments.py --make_confidence=conf_ref.npy --save_numbers=True`

By passing the `--make_confidence` flag, the experiment is run twice; first to generate the confidence-dataset distance pairs for the kernel smoother, and the second to generate the results for the experiment. 
The `--save_numbers` flag indicates that all of the results should be saved.
Different configurations can be used for the experiment by either passing the correct argument to `run_experiments.py` or by making edits to `config.py`.

To generate the essential plots for the paper, use the command:
`python prepare_plots.py --exp_dir=experiments/experiment_XXXXX`
where XXXXX is the experiment number for which the experiment results are saved. 

## Suplementary Experiments
### MLP Experiment
To obtain the results for the MLP experiment, use the command:

`python run_experiments.py --experiment=mlp --threshold_conf=True --make_confidence=conf_ref.npy --save_numbers=True`

Here the `--experiment=mlp` flag indicates that the experiment should use the MLP as a classifier and the `threshold_conf` flag indicates that ATC should be used to calculate confidence.
