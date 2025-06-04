import numpy as np
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from copy import deepcopy
import pandas as pd
import time
from functools import partial
from utils import tqdm_joblib, compute_pdists, gaussian, find_ATC_threshold
from tqdm import tqdm
import os

class BaseExperiment:
    """
    Class to be inherited for experiments. When inheriting, only need to finish train function and all other metrics desired. Train must return a model that when called returns the scores.
    """
    def __init__(self, name, dataset, experiment_dir, num_neighbors=5, kernel_width=0.05, explore=([''], ['']), confidence_ref='', save_numbers=False, make_confidence='', unique=True, threshold_conf=False, tidy=False):
        # Prune duplicate
        for task_data in dataset.all_tasks_data.values():
            if unique:
                # _, unq_inds = np.unique(task_data["id"], return_index=True)
                unq_inds = [task_data['input_sentence'].index(x) for x in set(task_data['input_sentence'])]
                task_data['input_sentence'] = [task_data['input_sentence'][ind] for ind in unq_inds]
                task_data['correctness'] = task_data['correctness'][unq_inds]
                task_data['embeddings'] = task_data["embeddings"][unq_inds]
                task_data['perplexity'] = task_data['perplexity'][unq_inds]
        self.dataset = dataset
        self.model_names = [m.split('/')[1] for m in dataset.models]
        self.best_point_model_idx, self.best_point_model = self.find_avg_best_datapoint_model()
        self.best_set_model_idx, self.best_set_model = self.find_avg_best_dataset_model()
        self.make_confidence=make_confidence
        self.num_neighbors = num_neighbors
        self.kernel_width = kernel_width

        # Set load reference for confidence if it exists
        self.threshold_confidence = threshold_conf
        if len(confidence_ref) > 0:
            confidence_ref = np.load(confidence_ref, allow_pickle=True).item()
            self.kernel_width = confidence_ref['kernel_width']
            self.confidence_ref = confidence_ref['confidence_distance_pairs'][0]
        else:
            self.confidence_ref = None

        # Set up data recording infrastructure
        self.experiment_name = name
        self.x_keys, self.y_keys = explore
        self.exp_dir = experiment_dir
        self.tidy = tidy

        # Set up dir to save numberss
        if save_numbers:
            os.mkdir(f'{self.exp_dir}/all_numbers/')
        self.save_numbers= save_numbers


    def estimate_dataset_confidence(self, confidence, distances, x):
        c_weights = gaussian(x - distances, 0, self.kernel_width)
        c_weights = c_weights / c_weights.sum()
        c_est = np.sum(confidence * c_weights[:, np.newaxis], axis=0)
        c_std = np.sqrt(np.sum(np.square(confidence-c_est) * c_weights[:, np.newaxis], axis=0))
        return c_est, c_std


    def stack_confidence_list(self, confidence, selected_task=None, r=None):
        dists, confs = [], []
        confidence = deepcopy(confidence)
        if r is not None:
            task_pairs_list = confidence[r]
        else:
            task_pairs_list = {}
            for ref_conf in confidence.values():
                for task, task_list in ref_conf.items():
                    if task not in task_pairs_list:
                        task_pairs_list[task] = task_list
                    else:
                        task_pairs_list[task] += task_list
        for task, task_list in task_pairs_list.items():
            if selected_task is not None and task == selected_task:
                continue
            for point in task_list:
                dists.append(point[0].mean())
                confs.append(point[1])

        dists = np.stack(dists)
        confs = np.stack(confs)
        return dists, confs


    def small_dataset(self, acc, emb, keep_rate=0.1, return_other_samps=False):
        nsamp = acc.shape[0]
        samps = np.random.choice(np.arange(nsamp), size=int(np.floor(nsamp*keep_rate)), replace=False)
        if not return_other_samps:
            return acc[samps, :], emb[samps, :]
        else:
            other_samps = np.setdiff1d(np.arange(nsamp), samps)
            return (acc[samps, :], emb[samps, :]), (acc[other_samps, :], emb[other_samps, :])


    def split_dataset(self, task_name, ref_ratio=0, dataset_rate=1):
        task_data = self.dataset.all_tasks_data[task_name]

        # Sample from the target task to seed the data
        nsamp = task_data['embeddings'].shape[0]
        ref_size = min(int(nsamp * ref_ratio), 50)
        ref_samps = np.random.choice(np.arange(nsamp), size=ref_size, replace=False)
        test_samps = np.setdiff1d(np.arange(nsamp), ref_samps)

        # Pull embeddings and correctness for target
        test_emb = task_data['embeddings'][test_samps, :]
        test_acc = task_data['correctness'][test_samps, :]

        # Pull perplexity for target
        test_perp = task_data['perplexity'][test_samps, :]
        
        # Pull embeddings and correctness for training
        train_emb = [task_data['embeddings'][ref_samps, :]]
        train_acc = [task_data['correctness'][ref_samps, :]]
        for other_task_name, other_data in self.dataset.all_tasks_data.items():
            if (other_task_name != task_name) and (other_task_name != 'raft:subset=one_stop_english'):
                train_emb.append(other_data["embeddings"])
                train_acc.append(other_data['correctness'])
        train_emb = np.vstack(train_emb)
        train_acc = np.vstack(train_acc)
        if dataset_rate < 1:
            return self.small_dataset(train_acc, train_emb, dataset_rate), self.small_dataset(test_acc, test_emb, dataset_rate)
        else:
            return (train_acc, train_emb), (test_acc, test_emb), test_perp


    def find_best_model_from_scores(self, scores, test_acc):
        mean_scores = np.mean(scores, axis=0)
        mean_test_acc = np.mean(test_acc, axis=0)
        model_idx = np.argsort(mean_scores)[-1]
        model_name = self.model_names[model_idx]
        model_acc = mean_test_acc[model_idx]
        return model_name, model_acc, model_idx


    def find_pointwise_model_acc_from_scores(self, scores, test_acc, train_acc):
        tie_breaker_scores = scores + 1e-2 * train_acc.mean(axis=0)
        model_indices = np.argsort(tie_breaker_scores, axis=1)[:, -1]
        model_acc = test_acc[np.arange(len(model_indices)), model_indices]
        return model_acc.mean()


    def find_aug_best_model(self, predictions, dataset_confidence, test_acc):
        aug_scores = predictions*dataset_confidence + (1-predictions)*(1-dataset_confidence)
        best_aug_model, best_aug_model_acc, best_aug_model_idx = self.find_best_model_from_scores(aug_scores, test_acc)
        best_set_model_acc = np.mean(test_acc, axis=0)[self.best_set_model_idx]
        if self.model_better_than_baseline(predictions, dataset_confidence, best_aug_model_idx, self.best_set_model_idx) < 0.6:
            best_aug_model, best_aug_model_acc, best_aug_model_idx = self.best_set_model, best_set_model_acc, self.best_set_model_idx
        if np.any(np.isnan(aug_scores)):
            raise Exception('Nan in Aug score')
        return best_aug_model, best_aug_model_acc, best_aug_model_idx, aug_scores

    
    def find_true_best_model(self, test_acc):
        mean_test_acc = np.mean(test_acc, axis=0)
        model_idx = np.argsort(mean_test_acc)[-1]
        model_name = self.model_names[model_idx]
        model_acc = mean_test_acc[model_idx]
        return model_name, model_acc, model_idx

    
    def find_avg_best_datapoint_model(self):
        all_acc = np.vstack([task['correctness'] for task in self.dataset.all_tasks_data.values()])
        avg_acc = np.mean(all_acc, axis=0)
        model_idx = np.argsort(avg_acc)[-1]
        model_name = self.model_names[model_idx]
        print('Best datapoint model on average is:', model_name)
        return model_idx, model_name


    def find_avg_best_dataset_model(self):
        all_acc = np.vstack([np.mean(task['correctness'], axis=0) for task in self.dataset.all_tasks_data.values()])
        avg_acc = np.mean(all_acc, axis=0)
        model_idx = np.argsort(avg_acc)[-1]
        model_name = self.model_names[model_idx]
        print('Best dataset model on average is:', model_name)
        return model_idx, model_name


    def find_best_perp_model(self, test_perp, test_acc):
        mean_scores = np.mean(test_perp, axis=0)
        mean_test_acc = np.mean(test_acc, axis=0)
        model_idx = np.argsort(mean_scores)[-1]
        model_name = self.model_names[model_idx]
        model_acc = mean_test_acc[model_idx]
        return model_name, model_acc, model_idx


    def find_point_dist(self, test_emb, train_emb, nn=-1):
        dists = compute_pdists(test_emb, train_emb, dis='Cosine')
        dists = np.maximum(dists, 1e-7) # Cast negative distances to zero
        if nn==-1:
            return np.mean(dists, axis=1)
        else:
            return np.sort(dists, axis=1)[:, 0:nn]
        

    def find_dataset_dist(self, task_name):
        dataset_emb = []
        for t, d in self.dataset.all_tasks_data.items():
            if t != task_name:
                dataset_emb.append(d['task_embedding'])
            else:
                task_emb = d['task_embedding']
        dists = compute_pdists(task_emb[np.newaxis, :], np.stack(dataset_emb), dis='Cosine')
        return np.sort(dists, axis=1)[0, 0]


    def augment_scores(self, scores, train_emb, test_emb, confidence_ref=None):
        pdists = self.find_point_dist(test_emb, train_emb, nn=self.num_neighbors)
        confidence_ref = confidence_ref if confidence_ref is not None else self.confidence_ref
        bins = np.arange(0, 1, 1/confidence_ref.shape[0])
        binplace = np.digitize(pdists, bins) - 1
        confidence = confidence_ref[binplace, :]
        return confidence*scores + (1-confidence)*(1-scores)


    def model_better_than_baseline(self, predictions, confidence, model_idx, baseline_idx, num_samples=1000):
        model_p = confidence[model_idx]
        baseline_p = confidence[baseline_idx]
        n_pos = predictions.sum(axis=0)
        n_neg = predictions.shape[0] - n_pos
        model_sample = np.random.binomial(n_pos[model_idx], model_p, size=(num_samples)) + np.random.binomial(n_neg[model_idx], 1-model_p, size=(num_samples))
        baseline_sample = np.random.binomial(n_pos[baseline_idx], baseline_p, size=(num_samples)) + np.random.binomial(n_neg[baseline_idx], 1-baseline_p, size=(num_samples))
        num_greater = model_sample > baseline_sample
        return num_greater.mean()


    def find_average_threshold_confidence(self, model, val_data, test_scores):
        val_scores = model(val_data[1])[0]
        val_entropy = val_scores * np.log(val_scores) + (1-val_scores) * np.log(1-val_scores)
        val_entropy = np.nan_to_num(val_entropy)
        val_conf = (val_scores > 0.5) == val_data[0]
        val_errors = 1 - val_conf.mean(axis=0)
        # t_idx = np.round(val_errors * val_scores.shape[0])
        # threshold = np.sort(val_entropy, axis=0)[t_idx.astype(int), np.arange(val_scores.shape[1])]
        threshold = np.stack([find_ATC_threshold(val_entropy[:, i], val_conf[:, i])[1] for i in range(val_data[0].shape[1])])
        test_entropy = test_scores * np.log(test_scores) + (1-test_scores) * np.log(1-test_scores)
        test_entropy = np.nan_to_num(test_entropy)
        return 1-np.mean(test_entropy < threshold, axis=0)


    def train(self, params, train_data):
        # To be implemented in inherited classes
        pass


    def evaluate(self, model, train_data, test_data, test_perp, ref_ratio=None, task_name=None, dataset_confidence=None, val_data=None):
        scores, metrics = model(test_data[1])
        predictions = scores > 0.5

        # Find our scored best model
        best_scored_model, best_scored_model_acc, best_scored_model_idx  = self.find_best_model_from_scores(predictions, test_data[0])

        # Find performance of absolute best model
        true_best_model, true_best_model_acc, true_best_model_idx = self.find_true_best_model(test_data[0])

        # Find lowest perplexity model
        perp_best_model, perp_best_model_acc, best_perp_idx = self.find_best_perp_model(test_perp, test_data[0])

        # Find acc if we treat each point as independent
        pointwise_model_acc = self.find_pointwise_model_acc_from_scores(scores, test_data[0], train_data[0])
        
        # Find performance of average best model
        best_point_model_acc = np.mean(test_data[0], axis=0)[self.best_point_model_idx]
        best_set_model_acc = np.mean(test_data[0], axis=0)[self.best_set_model_idx]

        # Find best model using naive confidence estimate
        naive_conf_model, naive_conf_model_acc, naive_conf_model_idx = self.find_best_model_from_scores(scores, test_data[0])

        # Find the dataset distance
        # dataset_dist = self.find_dataset_dist(task_name)
        dataset_dist = self.find_point_dist(test_data[1], train_data[1], nn=self.num_neighbors).mean(axis=0)

        # Find the confidence of prediction
        confidence = predictions==test_data[0]
        oracle_confidence = confidence.mean(axis=0)
        best_conf_model, best_conf_model_acc, best_conf_model_idx, oracle_conf_scores = self.find_aug_best_model(predictions, oracle_confidence, test_data[0])

        metrics['confidence'] = confidence.astype(int)
        metrics['naive_conf_scores'] = scores.mean(axis=0)
        metrics['pred_scores'] = predictions.mean(axis=0)
        metrics['oracle_conf_scores'] = oracle_conf_scores.mean(axis=0)
        metrics['perp_scores'] = test_perp.mean(axis=0)
        metrics['accs'] = test_data[0].mean(axis=0)
        metrics['dataset_dist'] = dataset_dist

        # Put model names in metrics
        metrics['scored_model'] = best_scored_model
        metrics['scored_model_idx'] = best_scored_model_idx
        metrics['naive_conf'] = naive_conf_model
        metrics['naive_conf_idx'] = naive_conf_model_idx
        metrics['best_model'] = true_best_model
        metrics['best_model_idx'] = true_best_model_idx
        metrics['perp_model'] = perp_best_model
        metrics['perp_model_idx'] = best_perp_idx
        metrics['dataset_model_idx'] = self.best_set_model_idx
        metrics['dataset_model'] = self.best_set_model
        metrics['conf_model_idx'] = best_conf_model_idx
        metrics['conf_model'] = best_conf_model

        length = np.shape(test_data[0])[0]
        metrics['task_length'] = length
        model_acc = {
            'scored_model_acc': best_scored_model_acc,
            'best_model_acc': true_best_model_acc,
            'perp_model_acc': perp_best_model_acc,
            'naive_conf_acc': naive_conf_model_acc,
            # 'datapoint_model_acc': best_point_model_acc,
            'dataset_model_acc': best_set_model_acc,
            # 'pointwise_model_acc': pointwise_model_acc,
            'conf_model_acc': best_conf_model_acc,
            }

        # Find our best augmented score model
        if self.confidence_ref is not None:
            confidence_ref = self.confidence_ref
            dists, confs = self.stack_confidence_list(confidence_ref, selected_task=task_name)
            dataset_confidence, _ = self.estimate_dataset_confidence(confs, dists, dataset_dist.mean())
            best_aug_model, best_aug_model_acc, best_aug_model_idx, aug_scores = self.find_aug_best_model(predictions, dataset_confidence, test_data[0])
            metrics['dataset_confidence'] = dataset_confidence
            metrics['aug_model'] = best_aug_model
            metrics['aug_scores'] = aug_scores.mean(axis=0)
            model_acc['aug_model_acc'] = best_aug_model_acc
            metrics['aug_model_idx'] = best_aug_model_idx

        if self.threshold_confidence:
            atc_confidence = self.find_average_threshold_confidence(model, val_data, scores)
            if atc_confidence[10] == 0:
                atc_confidence[10] = 1.
            best_atc_model, best_atc_model_acc, best_atc_model_idx, atc_scores = self.find_aug_best_model(predictions, atc_confidence, test_data[0])
            metrics['atc_confidence'] = atc_confidence
            metrics['atc_model'] = best_atc_model
            metrics['atc_scores'] = atc_scores.mean(axis=0)
            model_acc['atc_model_acc'] = best_atc_model_acc
            metrics['atc_model_idx'] = best_atc_model_idx

        return model_acc, metrics, length


    def train_and_eval_for_task(self, task_name, ref_ratio, params):
        train_data, test_data, test_perp = self.split_dataset(task_name, ref_ratio, params['dataset_rate'])
        if self.threshold_confidence:
            val_data, train_data = self.small_dataset(train_data[0], train_data[1], keep_rate=0.05, return_other_samps=True)
        else:
            val_data = None
        model = self.train(params, train_data)
        model_acc, metrics, length = self.evaluate(model, train_data, test_data, test_perp, ref_ratio=ref_ratio, task_name=task_name, val_data=val_data)

        # Add the other metrics
        metrics['confidence'] = metrics['confidence'].mean(axis=0)
        return model_acc, metrics, length


    def train_and_eval_for_split_data(self, ref_ratio, params, dataset_split=0.2):
        # Compile all data from tasks together
        all_acc = np.vstack([task['correctness'] for task in self.dataset.all_tasks_data.values()])
        all_emb = np.vstack([task['embeddings'] for task in self.dataset.all_tasks_data.values()])
        all_perp = np.vstack([task['perplexity'] for task in self.dataset.all_tasks_data.values()])
        all_numbers = {}
        if all_acc.shape[0] != all_emb.shape[0]:
            raise Exception
        if params['dataset_rate'] < 1:
            all_acc, all_emb = self.small_dataset(all_acc, all_emb, params['dataset_rate'])
        
        # Sample train and test data
        nsamp = all_acc.shape[0]
        test_samps = np.random.choice(np.arange(nsamp), size=int(np.floor(nsamp * dataset_split)), replace=False)
        train_samps = np.setdiff1d(np.arange(nsamp), test_samps)
        train_data = (all_acc[train_samps, :], all_emb[train_samps, :])
        test_data = (all_acc[test_samps, :], all_emb[test_samps, :])
        test_perp = all_perp[test_samps, :]

        # Train and eval
        model = self.train(params, train_data)
        model_acc, metrics, _ = self.evaluate(model, train_data, test_data, test_perp)
        if self.save_numbers:
            all_numbers['task_numbers'][task_name] = (model_acc, metrics)
        metrics['confidence'] = metrics['confidence'].mean(axis=0)

        # Save the numbers if needed
        if self.save_numbers:
            all_numbers['params'] = params
            all_numbers['dataset_split'] = dataset_split
            all_numbers['model_acc'] = model_acc
            all_numbers['metrics'] = metrics
            iteration_num = len(os.listdir(f'{self.exp_dir}/all_numbers/'))
            with open(f'{self.exp_dir}/all_numbers/{iteration_num:05}.pickle', 'wb') as f:
                pickle.dump(all_numbers, f, protocol=pickle.HIGHEST_PROTOCOL)
        return model_acc, {'All Datasets': metrics}, None


    def train_and_eval_for_all_tasks(self, ref_ratio, params):
        # Set up base confidence
        if len(self.make_confidence) > 0:
            ref_confs = {}
        else:
            ref_confs = None

        if self.save_numbers:
            all_numbers = {}
            all_numbers['task_numbers'] = {}
            all_numbers['params'] = params
            all_numbers['ref_ratio'] = round(ref_ratio, 2)

        # Run experiment for all tasks
        model_acc = {}
        metrics = {}
        lengths = []
        for task_name, task_data in self.dataset.all_tasks_data.items():
            task_model_acc, task_metrics, task_length = self.train_and_eval_for_task(task_name, ref_ratio, params)
            for k, v in task_model_acc.items():
                if k not in model_acc:
                    model_acc[k] = [v]
                else:
                    model_acc[k].append(v)
            metrics[task_name] = {}
            for k, v in task_metrics.items():
                metrics[task_name][k] = v
            lengths.append(task_length)

            if self.save_numbers:
                all_numbers['task_numbers'][task_name] = (task_model_acc, task_metrics)
            if self.make_confidence:
                ref_confs[task_name] = [task_metrics['dataset_dist'], task_metrics['confidence']]

        # Compute average statistics
        avg_acc = {}
        for model_acc_name in model_acc:
            model_name = model_acc_name[:-4]
            avg_acc[model_name] = np.array(model_acc[f'{model_name}_acc'])
            avg_acc[f'{model_name}_normalized'] = sum(np.array(model_acc[f'{model_name}_acc']).T * np.array(lengths)/sum(lengths))

        if self.save_numbers:
            all_numbers['overall_acc'] = avg_acc
            iteration_num = len(os.listdir(f'{self.exp_dir}/all_numbers/'))
            with open(f'{self.exp_dir}/all_numbers/{iteration_num:05}.pickle', 'wb') as f:
                pickle.dump(all_numbers, f, protocol=pickle.HIGHEST_PROTOCOL)
        return avg_acc, metrics, ref_confs


    def print_metrics(self, avg_acc, metrics):
        for k, v in avg_acc.items():
            print(f'Average {k} accuracy: {v}')
        for metric in list(metrics.values())[0].keys():
            for task_name in metrics.keys():
                print(metric, task_name, metrics[task_name][metric])


    def plot_accs(self, all_accs, ref_ratios, i_p, all_dists=None):
        captions = {
                'scored_model': 'Avg. acc with our choices', 
                'best_model': 'Avg. acc with best choices', 
                'datapoint_model': self.best_point_model, 
                'dataset_model': self.best_set_model, 
                'aug_model': 'Avg. acc with corrected choices', 
                'atc_model': 'Avg. acc with ATC correction',
                'perp_model': 'Lowest perplexity model', 
                'naive_conf': 'Avg. acc with naive confidence',
                'pointwise_model': 'Acc for model selected per point',
                'conf_model': 'Avg. acc with oracle confidence',
                }
        if self.experiment_name is None:
            raise Exception('No experiment name is provided')
        os.mkdir(f'{self.exp_dir}/task_specific_plots')
        for task_idx, task_name in enumerate(self.dataset.all_tasks_data):
            task_size = self.dataset.all_tasks_data[task_name]['correctness'].shape[0]
            plt.figure()
            plot_legend = []
            for k, v in all_accs.items():
                if 'normalized' not in k and 'point' not in k:
                    plt.errorbar(ref_ratios, np.mean(v[task_idx], axis=1), yerr=np.std(v[task_idx], axis=1))
                    plot_legend.append(captions[k])
            plt.legend(plot_legend,loc='lower right')
            plt.xlabel('In-distribution data points')
            plt.ylabel('Task Average Accuracy')
            plt.title(f'Accuracy for {task_size} points in {task_name}')
            plt.savefig(f'{self.exp_dir}/task_specific_plots/{task_idx}_{task_name[:10]}.png')
            plt.close()

        if all_dists is not None:
            os.mkdir(f'{self.exp_dir}/dist_plots')
            for k, v in all_accs.items():
                if 'normalized' not in k and ('naive' in k or 'aug' in k or 'scored' in k):
                    all_dists_fig = plt.figure()
                    for i_r, r in enumerate(ref_ratios):
                        plt.figure()
                        for task_idx, task_name in enumerate(all_dists):
                            plt.scatter(all_dists[task_name][i_r], np.mean(v[task_idx][i_r]), c='#1f77b4')
                        plt.xlabel('Dataset distance')
                        plt.ylabel(captions[k])
                        plt.title(f'Distance against {captions[k]} for {r:.2f} in-distribution ratio')
                        plt.savefig(f'{self.exp_dir}/dist_plots/{i_r}_{k}_plot.png')
                        plt.close()

                        plt.figure(all_dists_fig.number)
                        for task_idx, task_name in enumerate(all_dists):
                            plt.scatter(all_dists[task_name][i_r], np.mean(v[task_idx][i_r]), c='#1f77b4')
                    plt.xlabel('Dataset Distance')
                    plt.ylabel('Method Accuracy')
                    plt.title(f'{captions[k]} for all in-distribution ratios')
                    plt.savefig(f'{self.exp_dir}/dist_plots/all_plots_for_{k}')
                    plt.close()

        plt.figure()
        plot_legend = []
        for k, v in all_accs.items():
            task_mean_v = v.mean(axis=0)
            if 'normalized' not in k and 'datapoint' not in k:
                plt.errorbar(ref_ratios, np.mean(task_mean_v, axis=1), yerr=np.std(task_mean_v, axis=1))
                plot_legend.append(captions[k])
        plt.ylim(0.55, 0.80)
        plt.legend(plot_legend,loc='lower right')
        plt.xlabel('In-distribution data points')
        plt.ylabel('Average Accuracy')
        plt.title(f'Method accuracy for {self.experiment_name}')
        plt.savefig(f'{self.exp_dir}/dataset_ref.png')
        plt.close()

        plt.figure()
        plot_legend_normalized = []
        for k, v in all_accs.items():
            if 'normalized' in k and 'set' not in k:
                plt.errorbar(ref_ratios, np.mean(v, axis=1), yerr=np.std(v, axis=1))
                plot_legend_normalized.append(captions[k[:-11]])
        plt.legend(plot_legend_normalized,loc='lower right')
        plt.ylim(0.65, 0.80)
        plt.xlabel('In-distribution data points')
        plt.ylabel('Average Accuracy')
        plt.title(f'Data-point accuracy for {self.experiment_name}: {i_p}')
        plt.savefig(f'{self.exp_dir}/datapoint_ref.png')
        plt.close()



    def extract_param(self, keys, dictionary):
        if len(keys) > 1:
            return self.extract_param(keys[1:], dictionary[keys[0]])
        else:
            return dictionary[keys[0]]


    def replace_param(self, keys, dictionary, value):
        if len(keys) > 1:
            dictionary[keys[0]] = self.replace_param(keys[1:], dictionary[keys[0]], value)
            return dictionary
        else:
            dictionary[keys[0]] = value
            return dictionary


    def repack_params(self, params, param_length):
        if len(''.join(self.x_keys))==0 or len(''.join(self.y_keys))==0:
            return [params]
        all_params = []
        x_params = self.extract_param(self.x_keys, params)
        y_params = self.extract_param(self.y_keys, params)
        if not type(x_params) is list:
            x_params = [x_params*(i+1) for i in range(param_length)]
        if not type(y_params) is list:
            y_params = [y_params*(i+1) for i in range(param_length)]
        for x_l in x_params:
            for y_l in y_params:
                new_params = deepcopy(params)
                new_params = self.replace_param(self.x_keys, new_params, x_l)
                new_params = self.replace_param(self.y_keys, new_params, y_l)
                all_params.append(new_params)
        return all_params
        

    def run_experiment(self, ref_ratios, ite_n, params, param_length=10, dataset_split=None):
        all_params = self.repack_params(params, param_length)
        all_accs = []
        all_metrics = []

        if len(self.make_confidence) > 0:
            all_confs = {}
        
        # Flatten out all of the arguments for the experiment
        joblib_args = []
        joblib_idx = []
        for i_p, params in enumerate(all_params):
            all_accs.append({})
            all_metrics.append([])
            for i_r, r in enumerate(ref_ratios):
                all_metrics[i_p].append([])
                for ite in range(ite_n):
                    joblib_args.append((r, params))
                    joblib_idx.append((i_p, i_r, ite))
        
        # Define the experiment function based on dataset split
        train_func = self.train_and_eval_for_all_tasks if dataset_split is None else partial(self.train_and_eval_for_split_data, dataset_split=dataset_split)

        # Run the experiments in parallel
        n_jobs = 1 if len(joblib_args) < 10 else 5
        with tqdm_joblib(tqdm(desc="Experiments", total=len(joblib_args))) as progress_bar:
            for (i_p, i_r, ite), (avg_acc, metrics, ref_confs) in zip(joblib_idx, Parallel(n_jobs=n_jobs)(delayed(train_func)(r, params) for r, params in joblib_args)):
                print(f'\n Param set {i_p+1} out of {len(all_params)}')
                print(f'\n Iteration {ite+1}/{ite_n} for ratio {ref_ratios[i_r]}')
                if len(joblib_args) < 10:
                    self.print_metrics(avg_acc, metrics)
                for k, v in avg_acc.items():
                    for task_idx in range(len(self.dataset.all_tasks_data)):
                        if k not in all_accs[i_p]:
                            all_accs[i_p][k] = np.zeros((v.shape[0], len(ref_ratios), ite_n)) if len(v.shape) > 0 else np.zeros((len(ref_ratios), ite_n))
                        if len(all_accs[i_p][k].shape) < 3:
                            all_accs[i_p][k][i_r, ite] = v
                        else:
                            all_accs[i_p][k][task_idx, i_r, ite] = v[task_idx] 
                all_metrics[i_p][i_r].append(metrics)

                # Combine confidence list
                if len(self.make_confidence) > 0:
                    if (i_p, i_r) not in all_confs:
                        all_confs[(i_p, i_r)] = {t: [c] for t, c in ref_confs.items()}
                    else:
                        for t, conf in all_confs[(i_p, i_r)].items():
                            conf.append(ref_confs[t])

        if len(self.make_confidence) > 0:
            conf_reference = [{} for _ in all_params]
            for k in all_confs:
                conf_reference[k[0]][round(ref_ratios[k[1]], 2)] = all_confs[k]

            # Generate confidence graph against distance
            if not self.tidy:
                dists, confs = self.stack_confidence_list(conf_reference[0])
                plot_x, plot_y, plot_yerr = [], [], []
                for i in range(10):
                    x = i*0.1
                    c_est, c_std = self.estimate_dataset_confidence(confs, dists, x)
                    plot_x.append(x)
                    plot_y.append(c_est)
                    plot_yerr.append(c_std)
                plot_y = np.stack(plot_y)
                plot_yerr = np.stack(plot_yerr)
                for m_idx, m in enumerate(self.model_names):
                    fig = plt.figure()
                    plt.errorbar(plot_x, plot_y[:, m_idx], yerr=plot_yerr[:, m_idx])
                    plt.title(f'Fitted confidence plot for {m}')
                    plt.xlabel('Dataset distance')
                    plt.ylabel('Confidence of our method')
                    fig.savefig(f'{self.exp_dir}/kernel_confidnece_{m_idx}.png')
                    plt.close()


            np.save(self.make_confidence, {'kernel_width': self.kernel_width, 'params': all_params, 'confidence_distance_pairs': conf_reference})

        if len(ref_ratios) > 1 and len(self.make_confidence) == 0 and not self.tidy:
            for i_p, param_accs in enumerate(all_accs):
                all_dists = {task_name: [np.mean([metrics[task_name]['dataset_dist'].mean() for metrics in all_metrics[i_p][i_r]]) for i_r in range(len(ref_ratios))] for task_name in self.dataset.all_tasks_data}
                self.plot_accs(param_accs, ref_ratios, i_p, all_dists=all_dists)
        return all_accs
