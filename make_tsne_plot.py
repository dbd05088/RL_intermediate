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
np.random.seed(2)

good_samples = 'optimal_test.json'
with open(good_samples, 'r') as fp:
    optimal = json.load(fp)
good_samples = optimal['best_action']

results = np.load('results_tsne.npy')
#good_results = np.load('good_results_tsne.npy')

good_list = []
for i, list_ in enumerate(good_samples):
#    good_list.append(results[i*5000+np.array(list_), :])
    good_list.append(results[i*5000+np.random.randint(0, 5000, 50), :])
good_results = np.concatenate(good_list, axis=0)

width = 7.0 # 3.4: onecol, 7.0: twocol
height = width * 0.9

fig = plt.figure(figsize=(width, height))

label = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
color_list = ['#9CC3D5', '#DD4132', '#195190', '#B1624E', '#E0C568', '#CBCE91', '#3A6B35', '#EF9DAF', '#595959', '#FF8C00']

ax = plt.gca()

for i in range(10):
    ax.scatter(results[i*5000:(i+1)*5000, 0], results[i*5000:(i+1)*5000, 1], c=color_list[i], alpha=0.02, s=10)
    ax.scatter(good_results[i*50:(i+1)*50, 0], good_results[i*50:(i+1)*50, 1], c=color_list[i], s=10, label=label[i])
    #need = 50-len(np.unique(good_samples[i]))
    #if need > 0:
    #    additional = results[i*5000+np.random.randint(0, 5000, need)]
    #    ax.scatter(additional[:, 0], additional[:, 1], c=color_list[i], s=10)

plt.xticks([])
plt.yticks([])

ax.set_title('Random Policy', fontsize=16)

ax.legend(facecolor=(0.97, 0.97, 0.97))
plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=2.5)
plt.savefig('tsne_random.png')
