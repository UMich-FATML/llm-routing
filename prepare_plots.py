import numpy as np
from scipy import stats
import os
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from omegaconf import OmegaConf

captions = {
        'naive_conf': '$S_1$ eq. (3)',
        'scored_model': '$S_2$ eq. (4)',
        'aug_model': '$S_3$ eq. (8)',
        'conf_model': '$S_3$ true $p$',
        'perp_model': 'LL',
        'dataset_model': 'BMA',
        'best_model': 'Oracle',
        }
captions_list = list(captions.keys())
scores_length = len(captions)

colors = {
        'naive_conf': 'red',
        'scored_model': 'orange',
        'aug_model': 'hotpink',
        'conf_model': 'purple',
        'perp_model': 'black',
        'dataset_model': 'brown',
        'best_model': 'grey',
        }

markers = {
        'naive_conf': 'v',
        'scored_model': 'o',
        'aug_model': 's',
        'conf_model': 'p',
        'perp_model': 'P',
        'dataset_model': 'X',
        'best_model': 'D',
        }

linestyles = {
        'naive_conf': 'dashed',
        'scored_model': 'dotted',
        'aug_model': 'dashdot',
        'conf_model': (0, (5, 1)),
        'perp_model': (0, (3, 1, 1, 1)),
        'dataset_model': (0, (3, 1, 1, 1, 1, 1)),
        'best_model': (5, (10, 3)),
        }

model_sizes = {
        'google/flan-ul2': 20, 
        'salesforce/codegen-16b-mono': 16,
        'prakharz/dial-flant5-xl': 3,
        'tiiuae/falcon-40b': 40,
        'google/flan-t5-xl': 3, 
        'google/flan-t5-xxl': 11,
        'togethercomputer/gpt-jt-6b-v1': 6, 
        'eleutherai/gpt-neox-20b': 20,
        'ibm/mpt-7b-instruct': 7, 
        'bigscience/mt0-xxl': 13, 
        'google/ul2': 20,
        'meta-llama/llama-2-13b': 13,
        'meta-llama/llama-2-13b-chat': 13,
        'meta-llama/llama-2-13b-chat-beam': 13,
        'meta-llama/llama-2-70b': 70,
        'meta-llama/llama-2-70b-chat': 70,
        'meta-llama/llama-2-7b': 7,
        'meta-llama/llama-2-7b-chat': 7,
        'bigcode/starcoder': 15,
        }
model_sizes = {m.split('/')[-1]: v for m, v in model_sizes.items()}

all_data = np.load('generated_data.npy', allow_pickle=True).item()
used_models = [m.split('/')[-1] for m in all_data['models']]
del all_data['data']['blimp:phenomenon=binding,method=multiple_choice_separate_original,']
used_model_accs = np.array([v['correctness'].mean(axis=0) for v in all_data['data'].values()]).mean(axis=0)
used_data = {m: [model_sizes[m], used_model_accs[i_m]] for i_m, m in enumerate(used_models)}

helm_df = pd.DataFrame(used_data, index=['Model Size', 'Average Accuracy']).transpose()
helm_df['Average Accuracy'] = helm_df['Average Accuracy'].apply("{:.3f}".format)
helm_latex = helm_df.to_latex(escape=False, column_format='l' + 'c'*2)
print(helm_latex)

def fit_task_names(task_name):
    if 'raft' in task_name or 'entity' in task_name:
        task_name = task_name.split('=')[-1]
    elif 'mmlu' in task_name:
        task_name = task_name.split('=')[1]
    elif task_name == 'legal_support,method=multiple_choice_joint:':
        task_name = 'legal_support'
    else:
        task_name = task_name.split(':')[0]
    task_name = task_name.split('_')[0]
    return task_name


def parse_and_sort_data(numbers_dir, ite_n):
    all_task_metrics = {}
    all_task_accs = {}

    for pickle_file in os.listdir(numbers_dir):
        exp_data = np.load(os.path.join(numbers_dir, pickle_file), allow_pickle=True)
        if exp_data['ref_ratio'] not in all_task_accs:
            all_task_accs[round(exp_data['ref_ratio'], 2)] = {}
            all_task_metrics[round(exp_data['ref_ratio'], 2)] = {}

        task_accs = all_task_accs[round(exp_data['ref_ratio'], 2)]
        task_metrics = all_task_metrics[round(exp_data['ref_ratio'], 2)]

        for k, v in exp_data['task_numbers'].items():
            if k not in task_metrics:
                task_metrics[k] = {m_name: [] for m_name in v[1]}
                task_accs[k] = {a_name: [] for a_name in captions.keys()}
            for m_name in task_metrics[k]:
                task_metrics[k][m_name].append(v[1][m_name])
            for a_name in task_accs[k]:
                task_accs[k][a_name].append(v[0][a_name+'_acc'])

    ref_ratio_data = {}
    for r_ratio in all_task_metrics:
        task_metrics = all_task_metrics[r_ratio]
        task_accs = all_task_accs[r_ratio]

        acc_table = np.zeros((len(task_accs), scores_length, ite_n))
        for i_k, (k, v) in enumerate(task_accs.items()):
            for j_k, (a_k, a_v) in enumerate(v.items()):
                if a_k != captions_list[j_k]:
                    print(a_k, captions_list[j_k])
                    raise Exception
                acc_table[i_k, j_k, :] = np.array(a_v)
        acc_ratio_table = acc_table / acc_table[:, [-1], :]

        llama_count = {k: 0 for k in captions}
        param_counts = {k: [] for k in captions}
        true_conf, conf_mae = [], []
        strat_rank = {k: [] for k in captions}
        d_dists = np.zeros((len(task_metrics), ite_n))

        corr_table = np.zeros((len(task_metrics), scores_length, ite_n))
        rank_corr_table = np.zeros((len(task_metrics), scores_length, ite_n))
        for i_k, (k, v) in enumerate(task_metrics.items()):
            for j_k, metric in enumerate(['naive_conf_scores', 'pred_scores', 'aug_scores', 'oracle_conf_scores', 'perp_scores']):
                v[metric] = [np.nan_to_num(m_iter, neginf=-100) for m_iter in v[metric]]
                corrs = [np.corrcoef(m_ite, a_iter)[0, 1] for m_ite, a_iter in zip(v[metric], v['accs'])]
                rank_corrs = [stats.spearmanr(m_ite, a_iter).statistic for m_ite, a_iter in zip(v[metric], v['accs'])]
                corr_table[i_k, j_k, :] = corrs
                rank_corr_table[i_k, j_k, :] = rank_corrs
            for i_s, strat in enumerate(llama_count):
                if strat != captions_list[i_s]:
                    raise Exception
                for i_s, strat_model in enumerate(v[strat]):
                    if v['dataset_model'][i_s] == strat_model:
                        llama_count[strat] += 1
                    param_counts[strat].append(model_sizes[strat_model])
                acc_args = np.argsort(v['accs'], axis=1)
                strat_acc_rank = [acc_args.shape[1] - np.argwhere(args==strat_idx).item() for strat_idx, args in zip(v[strat+'_idx'], acc_args)]
                strat_rank[strat].append(strat_acc_rank)
            true_conf.append(v['confidence'])
            conf_mae.append(np.abs(np.array(v['confidence'])-np.array(v['dataset_confidence'])))
            d_dists[i_k, :] = np.array(v['dataset_dist']).mean(axis=-1)
        ref_ratio_data[r_ratio] = [acc_table, acc_ratio_table, corr_table, rank_corr_table, llama_count, param_counts, np.array(true_conf), np.array(conf_mae), strat_rank, d_dists]
    return ref_ratio_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Location of experiment to generate figures for')
    args = parser.parse_args()
    numbers_dir = os.path.join(args.exp_dir, 'all_numbers')
    config_dir = os.path.join(args.exp_dir, 'config.yaml')

    exp_config = OmegaConf.load(config_dir)
    ite_n = exp_config['ite_n']

    ref_ratio_data = parse_and_sort_data(numbers_dir, ite_n)

    try:
        os.mkdir('pictures')
    except:
        print('pictures directory already exists')
    try:
        os.mkdir('pictures/corr_dist')
    except:
        print('corr distance directory already exists')

    r_focus = 0
    acc_table, acc_ratio_table, corr_table, rank_corr_table, llama_count, param_counts, true_conf, _, strat_rank, d_dists = ref_ratio_data[r_focus]
    mean_acc = acc_table.mean(axis=0).mean(axis=-1)
    std_acc = acc_table.mean(axis=0).std(axis=-1)
    mean_corr = corr_table.mean(axis=0).mean(axis=-1)
    std_corr = corr_table.mean(axis=0).std(axis=-1)
    mean_rank_corr = rank_corr_table.mean(axis=0).mean(axis=-1)


    indices = [v for v in captions.values()]
    acc_df = pd.DataFrame(index=indices)
    std_df = pd.DataFrame(index=indices)

    acc_df['Accuracy'] = mean_acc
    std_df['Accuracy'] = std_acc
    acc_df['Acc. Ratio'] = acc_ratio_table.mean(axis=0).mean(axis=-1)
    acc_df['Correlation'] = mean_corr
    std_df['Correlation'] = std_corr
    acc_df['Spearman'] = mean_rank_corr 

    for c in ['Accuracy', 'Correlation', 'Spearman', 'Acc. Ratio']:
        idx_best = acc_df[c].idxmax()
        acc_df[c] = acc_df[c].apply("{:.3f}".format) #+ '$\pm$' + std_df[c].apply("{:.3f}".format)
        best_val = '\\textbf{'+acc_df[c][idx_best]+'}'
        # best_val = acc_df[c][idx_best].split('$\pm$')
        # best_val = '$\pm$'.join(['\\textbf{' + best_val[0] + '}', best_val[1]])
        acc_df[c][idx_best] = best_val
    acc_df['\% Llama Model'] = np.array([v / llama_count['dataset_model'] for v in llama_count.values()])
    acc_df['Average \# Params'] = np.array([np.mean(v) for v in param_counts.values()])
    acc_df['\% Llama Model'] = acc_df['\% Llama Model'].apply("{:.2f}".format)
    acc_df['Average \# Params'] = acc_df['Average \# Params'].apply("{:.1f}".format)
    acc_df['Avg. Rank'] = [np.array(v).mean() for v in strat_rank.values()]
    acc_df['Avg. Rank'] = acc_df['Avg. Rank'].apply("{:.3f}".format)
    latex = acc_df.to_latex(escape=False, column_format='l' + 'c'*len(captions))
    print(latex)

    acc_data = np.zeros((len(ref_ratio_data), scores_length, ite_n))
    corr_data = np.zeros((len(ref_ratio_data), scores_length, ite_n))
    rank_corr_data = np.zeros((len(ref_ratio_data), scores_length, ite_n))
    true_conf_data = np.zeros((len(ref_ratio_data), ite_n))
    conf_mae_data = np.zeros((len(ref_ratio_data), ite_n))
    ref_ratios = []

    corr_dist_figs = {k: plt.figure() for k in captions}
    for i_r, (r, r_data) in enumerate(ref_ratio_data.items()):
        r_acc_table, r_acc_ratio_table, r_corr_table, r_rank_corr_table, r_llama_count, r_param_count, r_true_conf, r_conf_mae, r_strat_rank, r_d_dists = r_data
        acc_data[i_r, :, :] = r_acc_table.mean(axis=0)
        corr_data[i_r, :,  :] = r_corr_table.mean(axis=0)
        rank_corr_data[i_r, :,  :] = r_rank_corr_table.mean(axis=0)
        true_conf_data[i_r, :] = r_true_conf.mean(axis=0).mean(axis=-1)
        conf_mae_data[i_r, :] = r_conf_mae.mean(axis=0).mean(axis=-1)
        ref_ratios.append(r)

        if r not in [0., 0.10, 0.20, 0.28]:
            continue
        for i_v, (k, v) in enumerate(captions.items()):
            plt.figure(corr_dist_figs[k].number)
            plt.errorbar(r_d_dists.mean(axis=1), r_corr_table[:, i_v, :].mean(axis=-1), xerr=r_d_dists.std(axis=1), yerr=r_corr_table[:, i_v, :].std(axis=-1), fmt="o", color=colors[k])

    for k, v in captions.items():
        plt.figure(corr_dist_figs[k].number)
        plt.xlabel(f'Dataset Distance $u(d’)$', fontsize='x-large')
        plt.ylabel(f'Pearson Corr({v}, Accs.)'.replace('8', '7'), fontsize='x-large')
        plt.ylim(0., 1.05)
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.grid(True)
        plt.savefig(f'pictures/corr_dist/corr_dist_{k}.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plot_legend_acc = []
    for i_v, (k, v) in enumerate(captions.items()):
        plt.errorbar(ref_ratios, acc_data[:, i_v, :].mean(axis=-1), yerr=acc_data[:, i_v, :].std(axis=-1), linestyle=linestyles[k], marker=markers[k], color=colors[k])
        plot_legend_acc.append(v)
    plt.xlabel('$\\alpha$', fontsize='x-large')
    plt.ylabel('Average Accuracy', fontsize='x-large')
    plt.legend(plot_legend_acc,loc='lower right')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid(True)
    plt.savefig('pictures/acc_ref_ratio.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plot_legend_corr = []
    for i_v, (k, v) in enumerate(captions.items()):
        if corr_data[:, i_v, :].mean(axis=-1).any() < 0.01:
            continue
        plt.errorbar(ref_ratios, corr_data[:, i_v, :].mean(axis=-1), yerr=corr_data[:, i_v, :].std(axis=-1), linestyle=linestyles[k], marker=markers[k], color=colors[k])
        plot_legend_corr.append(v.replace('8', '7'))
    plt.xlabel('$\\alpha$', fontsize='x-large')
    plt.ylabel('Pearson Corr(Score, Accs.)', fontsize='x-large')
    plt.legend(plot_legend_corr, loc='lower right')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid(True)
    plt.savefig('pictures/corr_ref_ratio.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plot_legend_rank_corr = []
    for i_v, (k, v) in enumerate(captions.items()):
        if rank_corr_data[:, i_v, :].mean(axis=-1).any() < 0.01:
            continue
        plt.errorbar(ref_ratios, rank_corr_data[:, i_v, :].mean(axis=-1), yerr=rank_corr_data[:, i_v, :].std(axis=-1), linestyle=linestyles[k], marker=markers[k], color=colors[k])
        plot_legend_rank_corr.append(v.replace('8', '7'))
    plt.xlabel('$\\alpha$', fontsize='x-large')
    plt.ylabel('Spearman Corr(Score, Accs.)', fontsize='x-large')
    plt.legend(plot_legend_rank_corr, loc='lower right')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid(True)
    plt.savefig('pictures/rank_corr_ref_ratio.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.errorbar(ref_ratios, true_conf_data.mean(axis=-1), yerr=true_conf_data.std(axis=-1), linestyle='--', marker='^', color='green')
    plt.xlabel('$\\alpha$', fontsize='x-large')
    plt.ylabel('Average Acc. of $g_m$s', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid(True)
    plt.savefig('pictures/true_conf_ref_ratio.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.errorbar(ref_ratios, conf_mae_data.mean(axis=-1), yerr=conf_mae_data.std(axis=-1), linestyle='--', marker='^', color='green')
    plt.xlabel('$\\alpha$', fontsize='x-large')
    plt.ylabel('MAE of estimating $p$', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid(True)
    plt.savefig('pictures/conf_mae_ref_ratio.pdf', bbox_inches='tight')
    plt.close()
