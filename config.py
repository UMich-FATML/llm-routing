from omegaconf import OmegaConf
import torch

dict_config = {
        'config': None,
        'data_files': ['generated_data.npy'],
        'dir': 'experiments',
        'save_numbers': False,
        'make_confidence': '',
        'experiment': 'nn',
        'dataset_split': None,
        'explore_param_A': '',
        'explore_param_B': '',
        'ref_min': 0,
        'ref_max': 0.30,
        'ref_step': 0.02,
        'ite_n': 10,
        'name': '',
        'confidence_ref': '',
        'threshold_conf': False,
        'tidy': False,
        'conf': {
            'kernel_width': 0.09,
            'num_neighbors': 19,
            },
        'params': {
            'nn': {'dataset_rate': 1, 'k_nn': 5},
            'mlp': {
                'dataset_rate': 1,
                'mid_dim': 1500,
                'num_epochs': 100,
                'lr': 1e-2,
                }
            }
        }

config = OmegaConf.create(dict_config)
