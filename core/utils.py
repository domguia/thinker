import torch
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
# sns.set_style('dark')

class MatchCount:
    def __init__(self, size):
        self.match_count = torch.zeros(size, dtype=int)
        self.max = 0
    def update(self, logits, targets):
        acc = (targets == torch.argmax(logits, dim=2)).sum(dim=1).cpu()
        uniques, counts = torch.unique(acc, return_counts=True)
        self.match_count[uniques] += counts
        self.max = uniques.max().item()
    def __str__(self):
        out = ''
        for c in self.match_count[:self.max]:
            out += f"{c:5d} "
        out += " max = %d\n" % (self.max,)
        mean = self.match_count / self.match_count.sum()
        for c in mean[:self.max]:
            out += f" {c:.2f} "
        return out
    def __repr__(self):
        return self.__dict__.__str__()

### test
# match_count = MatchCount(20)
# match_count.update(logits, targets)
# print(match_count)

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    def __call__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        args = [item.strip() for items in args for item in items.split(',')]
        self.__dict__.update(**{name: globals()[name] for name in args})
    def __str__(self):
        return self.__dict__.__str__()
    def __repr__(self):
        return self.__dict__.__repr__()

    
    # sample variable parameters
    def sample(self, param):
        if isinstance(param, str):
            param = self.__dict__[param]
        return CfgNode.sample_(param)

    @staticmethod
    def sample_(param):
        if isinstance(param, range):
            param = list(param)
        if isinstance(param, (list,set)):
            param = np.random.choice(param)
        return param


import pandas as pd
import seaborn as sns

def plot_loss_and_accuracy(logs, experiment=None, title=None):
    # Create a DataFrame from the list of dictionaries.
    if isinstance(logs, pd.DataFrame): df = logs
    else: df = pd.DataFrame(logs)

    # Filter current experiment
    if 'experiment' in df.columns:
        if experiment == None:
            experiment = df.iloc[-1].experiment
        df = df[df.experiment == experiment]

    # Plot the loss values on the left y-axis.
    sns.lineplot(data=df, x='iteration', y='loss', color='orange') #, hue='step')
    # sns.lineplot(data=df, x='iteration', y='probe_loss', color='peru') #, hue='step')
    plt.ylabel('Loss')
    # plt.yscale('log',base=2)

    # Plot the accuracy values on the right y-axis.
    plt.twinx()
    sns.lineplot(data=df, x='iteration', y='accuracy', color='purple') #, hue='latent')
    plt.ylabel('Accuracy')

    # Set the title of the plot.
    if title: plt.title(title)
    else: plt.title('Loss and Accuracy')
    plt.xlabel('Step')


    # Set the width of the plot.
    plt.gcf().set_size_inches(12, 4)

    # Show the plot.
    plt.show()

    #plot_hp_heatmap(logs)

def plot_hp_heatmap(logs, use_last_n_batch=200, aggr='max'):
    if isinstance(logs, pd.DataFrame): df = logs
    else: df = pd.DataFrame(logs)

    if len(df)<use_last_n_batch: return

    # Group the data by the latent and step columns.
    # df = df[-use_last_n_batch:].groupby(['latent', 'step'])
    df = df.groupby(['latent', 'step'])
    if aggr=='mean':
        df = df.mean()
    elif aggr=='max':
        df = df.max()
    else:
        assert f'Invalid argument aggr={aggr}'
    df = df.reset_index().pivot(index="latent", columns="step", values="accuracy")

    # if len(df)<2: return #not interresting
    # print('i')

    # Plot a heatmap of the average loss.
    sns.heatmap(df, annot=True)

    # Show the plot.
    plt.show()

# training utils
def loop_over_extremum(*args, corner=True, edge=True):
    # product('ABCD', 'xyz') --> Ax Bx Cx Dx   Az Bz Cz Dz

    # n = len(args)
    # for i in range(n):
    #     min = args[i][0]
    #     max = args[i][-1]
    # for ext in [min,max]:
    #     for j in range(n):
    #         if j != i:
    #             for val in args[j]:
    #                 yield (ext, val)
    n = len(args)
    args_ = [tuple(pool) for pool in args]
    maxs_ = [len(pool)-1 for pool in args]
    pools = [tuple(range(len(pool))) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        n_extremum = sum([ (p==0 or p==m) for m,p in zip(maxs_,prod)])
        if (corner and n==n_extremum) or (edge and (n-1)==n_extremum) :
            yield tuple([arg[p] for arg,p in zip(args,prod)])
    

