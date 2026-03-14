import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_loss_and_accuracy(logs, experiment=None, title=None, ax=None, y_log=False):
    # Create a DataFrame from the list of dictionaries.
    if isinstance(logs, pd.DataFrame): df = logs
    else: df = pd.DataFrame(logs)

    # Filter current experiment
    if 'experiment' in df.columns:
        if experiment is None:
            experiment = df.iloc[-1].experiment
        df = df[df.experiment == experiment]

    if ax is None: fig, ax = plt.subplots()

    # Plot the loss values on the left y-axis.
    sns.lineplot(data=df, x=df.index, y='loss', color='red', alpha=.1, ax=ax) #, hue='step')
    sns.lineplot(data=df, x=df.index, y=df.loss.rolling(100).mean(), label='loss', color='red', linewidth=.3, ax=ax)
    # if 'factor' in df.columns: # no that use full to visualize
    #     loss_factor = df.loss*df.factor
    #     sns.lineplot(data=df, x=df.index, y=loss_factor,  color='darkorange', alpha=.1, ax=ax)
    #     sns.lineplot(data=df, x=df.index, y=loss_factor.rolling(100).mean(), label='loss * weight', color='darkorange', ax=ax)
    # sns.lineplot(data=df, x='iteration', y='probe_loss', color='peru', ax=ax)
    ax.set_ylabel('Loss')
    if y_log: plt.yscale('log',base=2)

    # Plot the accuracy values on the right y-axis.
    ax2 = ax.twinx()
    sns.lineplot(data=df, x=df.index, y='accuracy', color='purple', alpha=.1, ax=ax2)
    sns.lineplot(data=df, x=df.index, y=df.accuracy.rolling(100).mean(), label='accuracy * weight', color='purple',  ax=ax2)
    if 'raw_acc' in df.columns:
        sns.lineplot(data=df, x=df.index, y='raw_acc',  color='violet',  alpha=.1, ax=ax2)
        sns.lineplot(data=df, x=df.index, y=df.raw_acc.rolling(100).mean(),  label='raw accuracy',  color='violet', linewidth=.3, ax=ax2)

    if 'factor' in df.columns:   sns.lineplot(data=df, x=df.index, y='factor',   label='difficulty weight',   color='gray', ax=ax2)

    # if 'loss_std' in df.columns: sns.lineplot(data=df, x=df.index, y='loss_std', label='loss_std', color='wheat', ax=ax2)
    # if 'acc_std' in df.columns:  sns.lineplot(data=df, x=df.index, y='acc_std',  label='acc_std',  color='lavender', ax=ax2)
      
    ax2.set_ylabel('Accuracy')

    best_loss = df.loss.min()
    best_acc = df.accuracy.max()

    # Set the title of the plot.
    if title: ax.set_title(title)
    else: ax.set_title(f"Min Loss {best_loss:.4f} / {best_acc:.2f} Highest Accuracy")
    ax.set_xlabel('Step')

    # # Set the width of the plot.
    ax.figure.set_size_inches(12, 4)
    # sns.move_legend(ax2, "lower right")
    ax.get_legend().set_visible(False)
    ax2.get_legend().set_visible(False)
    fig.legend(loc = "upper left") # "lower right"

def plot_hp_heatmap(logs, use_last_n_batch=200, aggr='max', ax=None):
    if isinstance(logs, pd.DataFrame): df = logs
    else: df = pd.DataFrame(logs)

    if len(df) < use_last_n_batch:
        return

    # Group the data by the latent and step columns.
    df = df.groupby(['latent', 'step'])
    if aggr == 'mean':
        df = df.mean()
    elif aggr == 'max':
        df = df.max()
    elif aggr == 'top10_avg':
        def avg_top_k(series, k):
            return series.nlargest(k).mean()
        df = df.agg(avg_top_k, k=2)
    else:
        assert f'Invalid argument aggr={aggr}'
    df = df.reset_index().pivot(index="latent", columns="step", values="accuracy")

    if ax is None: fig, ax = plt.subplots()

    # Plot a heatmap of the average loss.
    sns.heatmap(df, annot=True, ax=ax, annot_kws={"size": 12})

def compare_experiments(logs):
    if isinstance(logs, pd.DataFrame): df = logs
    else: df = pd.DataFrame(logs)
    
    unique_experiments = df.experiment.unique()
    num_experiments = len(unique_experiments)

    fig = plt.figure(figsize=(25, 15))

    # Create a 2x2 grid with the heatmap at the center
    grid = GridSpec(5, 4, figure=fig)
    heatmap_ax = plt.subplot(grid[1:-1, 1:-1])
    plot_hp_heatmap(df, aggr='max', ax=heatmap_ax)

    for i, exp_id in enumerate(unique_experiments):
        df_exp = df[df.experiment == exp_id]
        step, latent = int(df_exp.iloc[0].step), int(df_exp.iloc[0].latent)
        title = f"Loss and Accuracy w/ hyperparam = step: {step:.0f} | latent: {latent:.0f}"
        plot_loss_and_accuracy(df_exp, None, title=title, ax=plt.subplot(grid[latent-4, step-1]))


    # # Set the figure size for each subplot
    # for ax in [heatmap_ax] + loss_accuracy_axs:
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()




