import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl
import pickle

mpl.rcParams.update({
    'font.size'           : 8.0        ,
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

def get_results(f):
    lines = f.readlines()
    acc = []
    for line in lines:
        if 'accuracy' in line:
            list_ = line.split(' ')
            acc.append(float(list_[2][9:]))
    return acc

width = 3.4 # 3.4: onecol, 7.0: twocol
height = width * 0.9

fig = plt.figure(figsize=(width, height))

f = open('03-30 13:01:07.txt', 'r')
acc = get_results(f)
f = open('04-02 08:11:49.txt', 'r')
acc_high = get_results(f)

acc_seq = [np.array(acc[i*80:i*80+80]) for i in range(len(acc)//80)]
avg_acc = np.array([np.mean(accs) for accs in acc_seq])
std_acc = np.array([np.std(accs) for accs in acc_seq])

acc_high_seq = [np.array(acc_high[i*80:i*80+80]) for i in range(len(acc)//80)]
avg_high_acc = np.array([np.mean(accs) for accs in acc_high_seq])
std_high_acc = np.array([np.std(accs) for accs in acc_high_seq])

ax = plt.gca()
ax.plot([0, 6000], [(avg_acc[0]+avg_high_acc[0])/2]*2, ls='--', lw=1, c='k', label='Random Policy')
#ax.plot(np.arange(40, len(avg_acc)*80-39, 80), avg_acc, ls='-', c='#DD4132', label='One Step REINFORCE')
#ax.fill_between(np.arange(40, len(avg_acc)*80-39, 80), avg_acc-std_acc, avg_acc+std_acc, facecolor='#DD4132', alpha=0.2)
ax.plot(np.arange(40, len(avg_acc)*80-39, 80), avg_high_acc, ls='-', c='#195190', label='One Step REINFORCE')
ax.fill_between(np.arange(40, len(avg_high_acc)*80-39, 80), avg_high_acc-std_high_acc, avg_high_acc+std_high_acc, facecolor='#195190', alpha=0.2)

print((avg_acc[0]+avg_high_acc[0])/2)

ax.set_xlabel('Episode #', fontsize=10)
ax.set_ylabel('Reward', fontsize=10)
ax.set_xlim(0, 6000)
ax.set_ylim(0.52, 0.62)
ax.set_title('Test Accuracy as Reward', fontsize=12)

ax.legend(facecolor=(0.97, 0.97, 0.97), loc='lower right')

plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=2.5)
plt.savefig('result.png')