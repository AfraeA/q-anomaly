import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))


def retrieve_result_dfs(size=500, qIT_shots=1000, qRM_shots=8000, qRM_settings=8000, qVS_samples=50):
    cRBF_df = pd.read_csv(f'../Results/cRBF/dsize_{size}.csv', index_col=0)
    qIT_df = pd.read_csv(f'../Results/qIT/dsize_{size}_n_shots_{qIT_shots}.csv', index_col=0)
    qRM_df = pd.read_csv(f'../Results/qRM/dsize_{size}_n_shots_{qRM_shots}_n_settings_{qRM_settings}.csv', index_col=0)
    # TODO: Add other DFs
    return {'cRBF': cRBF_df, 'qIT': qIT_df, 'qRM':qRM_df}
def plot_performance_by_npc(smoothing=True, errorbar=True, alpha=0.2, size=500, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None):
    result_dfs = retrieve_result_dfs(size, qIT_shots, qRM_shots, qRM_settings, qVS_subsamples)
    metrics = np.array([['avgPrecision', 'auroc'], ['precision', 'recall']])
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    for kmethod, df in result_dfs.items():
        g_df = df.groupby('num_pc')
        for i,j in product(range(metrics.shape[0]), range(metrics.shape[1])):
            axes[i,j].set_ylim(0,1)
            if smoothing:
                metric_smoothed_mean = g_df.mean()[metrics[i,j]].ewm(alpha=0.2).mean()
            else:
                metric_smoothed_mean = g_df.mean()[metrics[i,j]]
            metric_smoothed_mean.plot(legend=True, label=kmethod, linestyle='dashed', marker='.', ax=axes[i,j], grid=True)
            if errorbar: 
                if smoothing:
                    metric_smoothed_max = g_df.max()[metrics[i,j]].ewm(alpha=0.2).mean()
                    metric_smoothed_min = g_df.min()[metrics[i,j]].ewm(alpha=0.2).mean()
                else: 
                    metric_smoothed_max = g_df.max()[metrics[i,j]]
                    metric_smoothed_min = g_df.min()[metrics[i,j]]    
                axes[i,j].fill_between(g_df['num_pc'].mean().values, metric_smoothed_min, metric_smoothed_max, alpha=0.2)
            axes[i,j].legend(loc='upper left')
    # Add code for Plots folder creation here
    plotsFolderName = f'{ROOT_DIR}/Plots/performance_by_npc'
    if not os.path.exists(plotsFolderName):
        os.makedirs(plotsFolderName)
    plotFileName += f'_size_{size}'
    plotFileName += f'_qIT_shots_{qIT_shots}'
    plotFileName += f'_qRM_shots_{qRM_shots}_qRM_settings_{qRM_settings}'
    plotFileName += f'_qVS_subsamples_{qVS_subsamples}'
    #TODO: add handling for hyperparameters of qDISC and qBBF
    plotFileName += '.png'
    fig.savefig(plotFileName)
    plt.close(fig)