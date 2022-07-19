import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl
import json

mpl.rcParams.update({
    'font.size'           : 10.0        ,
    'font.sans-serif'     : 'Arial'    ,
    'xtick.major.size'    : 2          ,
    'xtick.minor.size'    : 1.5        ,
    'xtick.major.width'   : 0.75       ,
    'xtick.minor.width'   : 0.75       ,
    'xtick.labelsize'     : 8.0        ,
    'xtick.direction'     : 'in'       ,
    'xtick.top'           : True       ,
    'ytick.major.size'    : 2          ,
    'ytick.minor.size'    : 1.5        ,
    'ytick.major.width'   : 0.75       ,
    'ytick.minor.width'   : 0.75       ,
    'ytick.labelsize'     : 8.0        ,
    'ytick.direction'     : 'in'       ,
    'xtick.major.pad'     : 2          ,
    'xtick.minor.pad'     : 2          ,
    'ytick.major.pad'     : 2          ,
    'ytick.minor.pad'     : 2          ,
    'ytick.right'         : True       ,
    'savefig.dpi'         : 600        ,
    'savefig.transparent' : True       ,
    'axes.linewidth'      : 0.75       ,
    'lines.linewidth'     : 1.0
})

good_samples = 'optimal_test.json'
with open(good_samples, 'r') as fp:
    optimal = json.load(fp)
good_samples = optimal['optimal_action']

results = np.load('results.npy')
results = 1-results
#good_results = np.load('good_results_tsne.npy')

good_list = []
for i, list_ in enumerate(good_samples):
    good_list.append(results[i*5000+np.array(list_)])
    #good_list.append(results[i * 5000 + np.random.randint(0, 5000, 50)])
good_results = np.concatenate(good_list, axis=0)

width = 7.0 # 3.4: onecol, 7.0: twocol
height = width * 0.9

fig = plt.figure(figsize=(width, height))

repeat_good = np.repeat(good_results, 100)

ax = plt.gca()

n_bins = np.linspace(0, 1, 13)-(1/24)
plt.hist([results, good_results], n_bins, weights=[np.ones(50000)/50000, np.ones(500)/500], label=['All Samples', 'Selected Samples'], color=['#195190', '#DD4132'])

ax.set_title('Test Acc as Reward, Greedified', fontsize=16)
ax.set_xlabel('Uncertainty', fontsize=10)
ax.set_ylabel('Relative Frequency', fontsize=10)

ax.legend(facecolor=(0.97, 0.97, 0.97))
plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=2.5)
plt.savefig('uncert_testreward_greedy.pdf')
