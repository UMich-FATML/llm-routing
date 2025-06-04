import matplotlib.pyplot as plt
import re
import string
from nltk.metrics.scores import f_measure
from scipy.spatial.distance import cdist
import numpy as np
import deepcopy

def normalize_text(text, articles=True):
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text)))) if articles else white_space_fix(remove_punc(lower(text)))


def quasi_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0

def f1_score(gold: str, pred: str) -> float:
    # The F1 score is for a group of data points, not per-instance
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret

def scatter(acc, scores, sel_id, best_id, names, data_name, fontsize=20):
    # Scatter function pulled from Misha's experiments
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(scores, acc)
    # for idx in [sel_id, best_id]:
        # ax.annotate('%0.2f' % acc[idx], (scores[idx], acc[idx]), xytext=(0, 5), textcoords='offset points', size=fontsize)
        # ax.annotate(names[idx], (scores[idx], acc[idx]), xytext=(0, 22), textcoords='offset points', size=fontsize)
    for idx in range(len(names)):
        # ax.annotate('%0.2f' % acc[idx], (scores[idx], acc[idx]), xytext=(0, 5), textcoords='offset points', size=fontsize)
        ax.annotate(names[idx], (scores[idx], acc[idx]), xytext=(0, 5), textcoords='offset points', size=fontsize)

        plt.title(data_name, fontsize=fontsize)
        plt.xlabel('model score', fontsize=fontsize)
        plt.ylabel('model accuracy', fontsize=fontsize)

        fig.savefig(f'rf_{data_name}.png')


def compute_pdists(x, y, dis="Euclidean"):
    if dis == "Euclidean":
        distance = cdist(x, y)
    elif dis == "Cosine":
        x = x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        y = y/np.linalg.norm(y, ord=2, axis=1, keepdims=True)
        distance = 1 - np.dot(x, y.T)
    return distance


import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def find_ATC_threshold(scores, labels): 
    sorted_idx = np.argsort(scores)
    
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
    fn = 0.0
    
    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
        if np.abs(fp - fn) < min_fp_fn: 
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]
    
    return min_fp_fn, thres


class ModelSelectionDataset:
    def __init__(self, ms_dataset):
        self.models = ms_dataset['models']
        self.num_models = ms_dataset['num_models']
        self.num_tasks = ms_dataset['num_tasks']
        self.all_tasks_data = ms_dataset['data']


    def save_data(self, filename='data.npy'):
        np.save(filename, {'models': self.models, 'num_models': self.num_models, 'num_tasks': self.num_tasks, 'data': self.all_tasks_data})

    
    @classmethod
    def merge_datasets(cls, a, b):
        if all([m_a != m_b for m_a, m_b in zip(a['models'], b['models'])]):
            models = a['models'] + b['models']
            if not all([task in b['data'] for task in a['data']]) and a['num_tasks'] != b['num_tasks']:
                raise Exception('Merging unequal models and unequal tasks')
            data = {}
            for task in a['data']:
                for p_a, p_b in zip(a['data'][task]['prompts'], b['data'][task]['prompts']):
                    if p_a != p_b:
                        raise Exception('Prompt mismatch')
                data[task] = deepcopy(a['data'][task])
                data[task]['predictions'] = a['data'][task]['predictions'] + b['data'][task]['predictions']
                data[task]['correctness'] = np.concatenate([a['data'][task]['correctness'], b['data'][task]['correctness']], axis=1)
                data[task]['perplexity'] = np.concatenate([a['data'][task]['perplexity'], b['data'][task]['perplexity']], axis=1)
            return {'models': models, 'num_models': len(models), 'num_tasks': a['num_tasks'], 'data': data}


    @classmethod
    def load_data(cls, filenames):
        if len(filenames)==1:
            dataset = np.load(filenames[0], allow_pickle=True).item()
        else:
            ms_datasets = [np.load(f_name, allow_pickle=True).item() for f_name in filenames]
            dataset = ms_datasets[0]
            for d_set in ms_datasets[1:]:
                dataset = cls.merge_datasets(dataset, d_set)
        return cls(dataset)


    @classmethod
    def load_from_prompts(cls, data_prompts, all_models):
        ms_dataset = {'models': all_models, 'num_models': len(all_models), 'num_tasks': len(data_prompts[2:]), 'data': {}}
        return cls(ms_dataset)
