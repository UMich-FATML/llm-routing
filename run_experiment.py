from omegaconf import OmegaConf
import argparse
import os
import numpy as np
from config import config
from nn_experiment import NearestNeighborsExperiment
from mlp_experiment import MultiLayerPerceptronExperiment
from utils import ModelSelectionDataset


if __name__ == '__main__':
    # Define arguments for command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='Base config to be used')
    parser.add_argument('--dir', default='experiments', type=str, help='Directory for results to be saved')
    parser.add_argument('--experiment', type=str, help='Which experiment to run')
    parser.add_argument('--save_numbers', type=bool, help='Detailed saving of all numbers')
    parser.add_argument('--make_confidence', type=str, help='Location to save confidence reference')
    parser.add_argument('--dataset_split', type=float, help='If None will split off of tasks otherwise split all datasets according to ratio')
    parser.add_argument('--explore_param_A', type=str, help='What parameters to explore')
    parser.add_argument('--explore_param_B', type=str, help='What parameters to explore')
    parser.add_argument('--name', help='Name the files will be saved under')
    parser.add_argument('--ite_n', type=int, help='How many times to repeat an experiment')
    parser.add_argument('--ref_min', type=float, help='The min reference ratio')
    parser.add_argument('--ref_max', type=float, help='The max reference ratio')
    parser.add_argument('--ref_step', type=float, help='How much the reference ratio changes between experiments')
    parser.add_argument('--confidence_ref', type=str, help='Reference file for using confidence')
    parser.add_argument('--threshold_conf', type=bool, help='Whether to use average threshold confidence. Warning: suggested to only use when k_nn is sufficiently large.')
    parser.add_argument('--tidy', type=bool, help='Delete any extraneous files made during the experiment')

    # Parse arguments and make config
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args['config'] == '':
        base_conf = config
    else:
        base_conf = OmegaConf.load(args['config'])
    args_conf = OmegaConf.create(args)
    config = OmegaConf.merge(base_conf, args_conf)

    # Make experiment directory and copy over nessecary files:
    try:
        os.mkdir(config['dir'])
    except:
        print('Directory to save experiments already exists!')
    if len(config['name']) > 0:
        exp_dir = f"{config['dir']}/{config['name']}"
        name = config['name']
    else:
        exp_num = len(os.listdir('experiments'))
        name = f'experiment_{exp_num:05}'
        exp_dir = f"{config['dir']}/{name}"
    try:
        os.mkdir(exp_dir)
    except:
        print('Warning: experiment directory already exists!')
    OmegaConf.save(config=config, f=f'{exp_dir}/config.yaml')

    dataset = ModelSelectionDataset.load_data(config['data_files'])
    del dataset.all_tasks_data['blimp:phenomenon=binding,method=multiple_choice_separate_original,']
    
    # Specify all relevent parameters
    ref_ratios = np.arange(config['ref_min'], config['ref_max'], config['ref_step'])
    explore_params = (config['explore_param_A'].split(' '), config['explore_param_B'].split(' '))
    if len(config['make_confidence']) > 0:
        conf_dir = os.path.join(exp_dir, config['make_confidence'])
    else:
        conf_dir = ''

    # Specify expeirment type
    experiment_type = {'nn': NearestNeighborsExperiment, 'mlp': MultiLayerPerceptronExperiment}[config['experiment']]
    params = config['params'][config['experiment']] 
    
    # Run experiment
    experiment = experiment_type(name, dataset, exp_dir, explore=explore_params, num_neighbors=config['conf']['num_neighbors'], kernel_width=config['conf']['kernel_width'], confidence_ref=config['confidence_ref'], save_numbers=(len(config['make_confidence'])==0 and config['save_numbers']), make_confidence=conf_dir, threshold_conf=config['threshold_conf'], tidy=config['tidy'])
    all_metrics = experiment.run_experiment(ref_ratios, config['ite_n'], dict(params), dataset_split=config['dataset_split'])

    # Run a second time if confidence reference is made
    if len(config['make_confidence']) > 0:
        experiment = experiment_type(name, dataset, exp_dir, explore=explore_params, num_neighbors=config['conf']['num_neighbors'], kernel_width=config['conf']['kernel_width'], confidence_ref=conf_dir, save_numbers=config['save_numbers'], threshold_conf=config['threshold_conf'], tidy=config['tidy'])
        new_metrics = experiment.run_experiment(ref_ratios, config['ite_n'], dict(params), dataset_split=config['dataset_split'])
    print('Finished with:', exp_dir)
