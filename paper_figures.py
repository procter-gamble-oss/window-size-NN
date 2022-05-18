# Copyright 2022 The Procter & Gamble Company
#
# Licenced under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the original License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Script to generate reproductible figures for the paper.
"""
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd

from data_lorenz import generate_1d_lorenz
from data_nilm import load_nilm_mapping
from data_paf import load_paf_rr

plt.style.use("tableau-colorblind10")
OUT_FOLDER = "output"


def plot_save(in_data, out_data, filename, xlabel='', ylabel=''):
    fig, ax = plt.subplots(dpi=200, figsize=(5, 2.5))
    ax.plot(in_data, label='input')
    ax.plot(out_data, label='output')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    fig.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.17)
    print("Saving {}.".format(filename))
    fig.savefig(os.path.join(OUT_FOLDER, filename))


def plot_lorenz_data(n_pts=250, noise_pct=10, sample_s=0.13):
    noisy_lorenz = generate_1d_lorenz(n_pts, noise_pct=noise_pct, tmstep=sample_s)
    predict = np.roll(noisy_lorenz, -1)
    predict[-1] = noisy_lorenz[-1]
    plot_save(noisy_lorenz, predict, 'lorenz-data.png', "Data index", "X-coordinate")


def plot_nilm_data(sample_s=7, home='REFIT-19', app='washing machine', start='2014-06-02T08:40', end='2014-06-02T10:15'):
    all_data = load_nilm_mapping(sample_s=sample_s)
    mains_data = all_data[home].loc[start:end,'mains'].values
    app_data = all_data[home].loc[start:end,app].values
    plot_save(mains_data, app_data, 'nilm-data.png', "Data index", "Active power (W)")


def plot_paf_data(id=3, start=5700, end=6000):
    all_data = load_paf_rr(['p'], id_range=[id,])
    id = 'p{:02d}'.format(id)
    hrv_data = all_data[id].loc[start:end,'RR_ms']
    labels = all_data[id].loc[start:end,'label'].map({ 'before PAF': 0, 'PAF': 1300, })
    plot_save(hrv_data, labels, 'paf-data.png', "R-R index", "R-R interval (ms)")


def create_bar_chart(data, dpi=200, figsize=(5, 1.8), bar_width=0.4):
    bbox = { 'boxstyle': 'square,pad=0',
             'facecolor': 'w', 'edgecolor': None, 'linewidth': 0, }
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    data.plot.bar(ax=ax, rot=0, width=bar_width)
    i_max = data.shape[1] if isinstance(data, pd.DataFrame) else 1
    for i in range(i_max):
        labels = ax.containers[i].datavalues
        labels = [ '' if l == 0 else str(l) for l in labels ]
        ax.bar_label(ax.containers[i], labels, label_type='edge', bbox=bbox)

    ax.margins(y=0.15)
    return fig, ax, bbox


def plot_win_sizes(result_df):
    for title, results in result_df.groupby('task'):
        results = results.dropna(subset=['method']).set_index('method')
        res = results.loc[np.logical_not(results['state_of_art']),'winsz_pts']
        res = res.sort_values()
        stoa = results.loc[results['state_of_art'],'winsz_pts']
        fig, ax, bbox = create_bar_chart(res)
        ax.set_ylabel("Sequence length (pts)")
        v_old = 0
        for label, val in stoa.items():
            ax.axhline(val, color='C1')
            offset = 120 if v_old > 0 and abs(val - v_old) < 120 else 1
            ax.annotate(label, (0.5, val + offset), ha='center', va='center', bbox=bbox)
            v_old = val + offset

        fig.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.17)
        filename = '{}-windows.png'.format(title)
        filepath = os.path.join(OUT_FOLDER, filename)
        print("Saving {}.".format(filepath))
        fig.savefig(filepath)


def combine_winsz_index(df, winsz_col='winsz_pts', sep='\n'):
    winsz = df.pop(winsz_col)
    df.index = [ '{:d}{}{}'.format(int(winsz[k]), sep, k) for k in df.index ]


def explore_results(result_df):
    plt.close('all')
    paf_all = result_df[result_df['task'] == 'paf']
    paf_all.pop('task')
    paf_all = paf_all.dropna(axis=1, how='all').dropna()
    paf_all.set_index('method', inplace=True)
    # Figure 1: bar chart with horizontal lines
    paf_stoa = paf_all[paf_all['state_of_art']]
    paf_res = paf_all[np.logical_not(paf_all['state_of_art'])]
    combine_winsz_index(paf_res)
    fig1, ax1, bbox = create_bar_chart(paf_res['roc_auc'])
    v_old = 0
    for label, row in paf_stoa.iterrows():
        val = row['roc_auc']
        winsz = int(row['winsz_pts'])
        ax1.axhline(val, color='C1')
        yl = max([val, v_old + 0.015])
        ax1.annotate(label, (0.5, yl), ha='center', va='center', bbox=bbox)
        v_old = val

    fig1.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
    print(paf_res)
    print(paf_stoa)
    # Figure 2: bar chart with different colors
    paf_stoa_info = paf_all.pop('state_of_art')
    combine_winsz_index(paf_all)
    fig2, ax2, bbox = create_bar_chart(paf_all)
    fig2.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
    print(paf_all)
    # Figure 3: 2D scatter plot in metric space
    fig3, ax3 = plt.subplots(dpi=200, figsize=(5, 1.8))
    colors = np.where(paf_stoa_info,'C1', 'C0')
    paf_all.plot.scatter(x='f1score', y='roc_auc', c=colors, ax=ax3)
    ax3.axhline(0.5, color='gray', alpha=0.5)
    ax3.axvline(0.5, color='gray', alpha=0.5)
    ax3.set_xlim(0, 1.1)
    ax3.set_ylim(0, 1.1)
    fig3.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
    # Figure 4: Bar chart per method
    known = result_df.dropna(subset=['method']).copy()
    known.loc[known['task'] == 'lorenz','perf'] = known['rel_err']
    nilm_tasks = ['nilm-kt', 'nilm-dw', 'nilm-wm']
    known.loc[known['task'].isin(nilm_tasks),'perf'] = known['mae']
    known.loc[known['task'] == 'paf','perf'] = known['roc_auc']
    stoa_info = known.pop('state_of_art')
    res_all = known[np.logical_not(stoa_info)]
    res_all = res_all.pivot('method', 'task', 'perf').dropna()
    fig4, ax4, bbox = create_bar_chart(res_all)
    fig4.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
    print(res_all)
    # Figure 5: F1-score = f(window size) and parabolic fitting
    score = result_df[['winsz_pts', 'task', 'state_of_art', 'f1score']]
    score = score.dropna()
    cmap = {'lorenz': 'C0', 'nilm-kt': 'C1', 'nilm-dw': 'C2', 'nilm-wm': 'C3',
            'paf': 'C4'}
    fig5, ax5 = plt.subplots(dpi=200, figsize=(5, 1.8))
    for task, tdf in score.groupby('task'):
        tdf.plot.scatter(x='winsz_pts', y='f1score', c=cmap[task], label=task, ax=ax5)
    ax5.set_ylim(0, 1)
    ax5.legend(loc='upper right')
    fig5.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
    print(score)
    plt.show()


def plot_winsz_metrics(result_df, metrics=['val_loss', 'f1score'],
tasks=['nilm-kt', 'nilm-dw', 'nilm-wm', 'paf']):
    plt.close('all')
    score = result_df[['winsz_pts', 'task'] + metrics]
    score = score[score['task'].isin(tasks)]
    cmap = dict(zip(tasks, [ 'C{}'.format(i) for i in range(len(tasks)) ]))
    for mname in metrics:
        fig, ax = plt.subplots(dpi=200, figsize=(5, 2.5))
        for task, tdf in score.groupby('task'):
            tdf.plot.scatter(x='winsz_pts', y=mname, c=cmap[task],
                            label=task.upper(), ax=ax)

        ax.legend(loc='upper left', fontsize='x-small')
        ax.set_xlabel('Sequence length (pts)')
        ax.set_xscale("log", nonpositive='clip')
        fig.subplots_adjust(left=0.14, top=0.95, right=0.95, bottom=0.19)
        filename = '{}-winsz.png'.format(mname)
        filepath = os.path.join(OUT_FOLDER, filename)
        print("Saving {}.".format(filepath))
        fig.savefig(filepath)


def plot_predictions(res_files=['network_seq2point-24-0.csv',
'network_seq2point-111-0.csv'], xmin=16000, xmax=18000, shift=43):
    plt.close('all')
    for i, f in enumerate(res_files):
        fdat = pd.read_csv(os.path.join(OUT_FOLDER, f))
        fig, ax = plt.subplots(dpi=200, figsize=(5, 1.8))
        fdat = fdat.loc[xmin-i*shift:xmax-i*shift,['y_true', 'y_pred']]
        fdat = fdat.reset_index(drop=True)
        fdat.plot(ax=ax)
        ax.legend(loc='upper right', fontsize='x-small')
        ax.set_xlabel('Point index')
        ax.set_ylabel('Normalized power')
        ax.set_ylim(-1, 16)
        fig.subplots_adjust(left=0.12, top=0.95, right=0.95, bottom=0.23)
        filename = f.rsplit('.')[0] + '.png'
        filepath = os.path.join(OUT_FOLDER, filename)
        print("Saving {}.".format(filepath))
        fig.savefig(filepath)


def print_winsz_sensitivity(result_df, metrics=['mae', 'rel_err', 'f1score',
'roc_auc']):
    for task, tdf in result_df.groupby('task'):
        print(task)
        mean_score = tdf[metrics].mean()
        min_score = tdf[metrics].min()
        max_score = tdf[metrics].max()
        summary = pd.DataFrame({
                'mean': mean_score,
                'min': min_score,
                'max': max_score,
                'dev_min': (mean_score - min_score) / mean_score,
                'dev_max': (max_score - mean_score) / mean_score,
                'range': max_score - min_score,
        })
        print(summary)
        print()


if __name__ == '__main__':
    plot_lorenz_data()
    plot_nilm_data()
    plot_paf_data()
    results = pd.read_csv(os.path.join(OUT_FOLDER, 'results-summary.csv'))
    plot_win_sizes(results)
    #explore_results(results)
    plot_winsz_metrics(results)
    plot_predictions()
    print_winsz_sensitivity(results)

