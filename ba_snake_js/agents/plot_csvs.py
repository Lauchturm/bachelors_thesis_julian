import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import statistics


def plot_h_fun_csv():
    # fname = "direction_h_fun_values.csv"
    # fname = "locomotion_h_fun_values.csv"
    # fname = 'script_h_fun_values.csv'
    fname = "direction_h_fun_values.csv"
    fname2 = "locomotion_h_fun_values.csv"
    # df = pd.read_csv(fname, sep=';', index_col=0)
    df = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    df = df[60:360]
    df2 = df2[60:]
    # print(type(df))
    # print(df)
    # ts = pd.Series(df)
    # print(type(ts))
    # print(ts)
    # df = df.drop(df[df['Value'] < 200 or df['Value'] > - 200].index)
    # df = df.where(200 > df['Value'] > -200)
    # series = pd.Series(df)
    # df = df[200 > df['Value'] > -200]
    # df = df[200 > df > -200]
    # df = df.ix[200 > df['Value'] > -200]
    # df = df.ix[200 > df > -200]
    # df = df.loc[200 > df.Value > -200, :]
    # df = df.loc[200 > df['Value'] > -200, :]
    # df = df[df[df > -200].any(axis=1)]
    # df = df[df[df < 200].any(axis=1)]
    df = df[df[df > -200].any(axis=1)]
    df = df[df[df < 200].any(axis=1)]
    df2 = df2[df2[df2 > -200].any(axis=1)]
    df2 = df2[df2[df2 < 200].any(axis=1)]
    # print(df)
    # print(series)
    # df2=df.reindex()
    # print(df2)
    # df = pd.Series(df['Value'].values[40:])

    # ax = sns.lineplot(x=df[0], y=df[1], sort=False, label='Snake')
    # ax = sns.lineplot(data=df, sort=False, label='Snake')
    # ax = sns.lineplot(data=df2, sort=False, legend=False, linewidth=1.5, color='red', label='Locomotion')
    # ax = sns.lineplot(data=df2, color='red', label='Locomotion')
    # ax2 = sns.lineplot(data=df, sort=False, legend=False, label='Direction', color='green', ax=ax)
    # ax.set(xlabel='timestep', ylabel='mean derivative')
    # ax.legend()
    plt.plot(df)
    plt.plot(df2)
    plt.legend(['direction', 'locomotion'])
    plt.show()
    # fig = ax.get_figure()
    # fig.savefig('myplot.pdf', dpi=300, bbox_inches='tight')
    # fig.savefig('myplot.png', dpi=300, bbox_inches='tight')


# def plot_bars():
#     N = 1
#     men_means = (20)
#     men_std = (2)
#
#     ind = np.arange(N)  # the x locations for the groups
#     width = 0.35  # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
#
#     women_means = (25)
#     women_std = (3)
#     rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
#
#     # add some text for labels, title and axes ticks
#     ax.set_ylabel('Scores')
#     ax.set_title('Scores by group and gender')
#     ax.set_xticks(ind + width / 2)
#     ax.set_xticklabels(('G1'))
#
#     ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
#     plt.show()


def var_stddev():
    # fname = "tracking_accuracies.csv"
    fname = "tracking_accuracies_train.csv"
    tracking = pd.read_csv(fname, sep=';')

    directs = tracking[tracking['control'] == 'Direct']
    print(directs)

    stats1 = pd.DataFrame()
    stats1["mean"] = directs.mean()
    stats1["Std.Dev"] = directs.std()
    stats1["Var"] = directs.var()
    print(stats1.T)

    indirects = tracking[tracking['control'] == 'Indirect']
    print(indirects)

    stats2 = pd.DataFrame()
    stats2["mean"] = indirects.mean()
    stats2["Std.Dev"] = indirects.std()
    stats2["Var"] = indirects.var()
    print(stats2.T)


def plot_bars2():
    # tips = sns.load_dataset("tips")
    # print(tips)
    fname = "tracking_accuracies.csv"
    # fname = "tracking_accuracies_train.csv"
    tracking = pd.read_csv(fname, sep=';')
    print(tracking)
    # ax = sns.barplot(x="day", y="tip", data=tips, capsize=.2)
    ax = sns.barplot(x="control", y="tracking success", data=tracking, capsize=.15, errwidth=2.5)
    ax.set(ylabel='Tracking success in %', xlabel='Evaluation Scenario')
    plt.ylim(40, 100)
    plt.show()

    fig = ax.get_figure()

    # pdfpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "barplot_presentation.pdf"))
    pngpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "barplot_presentation2.png"))
    # print(pdfpath)

    # fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')


def plot_bars_presentation():
    fname = "tracking_accuracies.csv"
    fname2 = "tracking_accuracies_train.csv"
    tracking = pd.read_csv(fname, sep=';')
    tracking2 = pd.read_csv(fname2, sep=';')
    print(tracking)
    print(tracking2)

    plt.style.use('seaborn-darkgrid')
    plt.subplot(121)  #, sharex='col', sharey='row')
    # fig, ax = plt.subplot(nrows=1, ncols=2,)  #, sharex='col', sharey='row')
    ax1 = sns.barplot(x="control", y="tracking success", data=tracking2, capsize=.15, errwidth=2.5, label='Training Scenario')
    plt.title('Training Scenario')
    plt.subplot(122)
    ax2 = sns.barplot(x="control", y="tracking success", data=tracking, capsize=.15, errwidth=2.5, label='Evaluation Scenario')
    plt.tick_params(labelleft='off')
    plt.title('Evaluation Scenario')

#    ax1.set(ylabel='Tracking success in %', xlabel='Training Scenario')
#    ax2.set(ylabel='Tracking success in %', xlabel='Evaluation Scenario')
    plt.ylim(40, 100)
    plt.show()


def plot_training_reward():
    fname = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                      "direction-v1-2-EpRewMean.csv"))
    fname2 = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                       "locomotion-v1-2-EpRewMean.csv"))
    # fname = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\direction-v1-2-EpRewMean.csv"
    # fname2 = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\locomotion-v1-2-EpRewMean.csv"
    df1 = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    print(df1)
    ax = sns.lineplot(x='Step', y='Value', data=df1, label='Indirect')
    ax2 = sns.lineplot(x='Step', y='Value', data=df2, label='Direct')
    # plt.plot(df1)
    # plt.plot(df2)
    ax.set(ylabel='Mean reward of the last 100 episodes', xlabel='Iteration')
    ax.legend()
    # plt.ylim(40, 100)
    plt.show()

    fig = ax.get_figure()
    pdfpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "rewardplot.pdf"))
    pngpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "rewardplot.png"))
    fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')
    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\rewardplot.pdf', dpi=300, bbox_inches='tight')
    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\rewardplot.png', dpi=300, bbox_inches='tight')


def plot_training_eplen():
    fname = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                      "direction-v1-2-EpLenMean.csv"))
    fname2 = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                       "locomotion-v1-2-EpLenMean.csv"))

    # fname = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\direction-v1-2-EpLenMean.csv"
    # fname2 = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\locomotion-v1-2-EpLenMean.csv"
    df1 = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    print(df1)
    ax = sns.lineplot(x='Step', y='Value', data=df1, label='Indirect')
    ax2 = sns.lineplot(x='Step', y='Value', data=df2, label='Direct')
    # plt.plot(df1)
    # plt.plot(df2)
    ax.set(ylabel='Mean length of the last 100 episodes', xlabel='Iteration')
    ax.legend()
    # plt.ylim(40, 100)
    plt.show()

    fig = ax.get_figure()
    pdfpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "eplenplot.pdf"))
    pngpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "eplenplot.png"))
    fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')

    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\eplenplot.pdf', dpi=300, bbox_inches='tight')
    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\eplenplot.png', dpi=300, bbox_inches='tight')


def plot_ep_this_iter():
    fname = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                      "direction-v1-2-EpThisIter.csv"))
    fname2 = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data",
                                       "locomotion-v1-2-EpThisIter.csv"))

    # fname = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\direction-v1-2-EpThisIter.csv"
    # fname2 = r"E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\locomotion-v1-2-EpThisIter.csv"
    df1 = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    print(df1)
    ax = sns.lineplot(x='Step', y='Value', data=df1, label='Indirect')
    ax2 = sns.lineplot(x='Step', y='Value', data=df2, label='Direct')
    # plt.plot(df1)
    # plt.plot(df2)
    ax.set(ylabel='Episodes in this iteration', xlabel='Iteration')
    ax.legend()
    # plt.ylim(40, 100)
    plt.show()

    fig = ax.get_figure()
    pdfpath = str(
        pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "epthisiterplot.pdf"))
    pngpath = str(
        pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent, "plots", "training_data", "epthisiterplot.png"))
    fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')

    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\epthisiterplot.pdf', dpi=300, bbox_inches='tight')
    # fig.savefig(r'E:\GITSTUFF\bachelor_thesis_snake\plots\training_data\epthisiterplot.png', dpi=300, bbox_inches='tight')


def plot_trajectory_circle():
    fname = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                      "plots", "circle-09-26", "indirect", "positions_4.csv"))
    fname2 = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                       "plots", "circle-09-26", "direct", "positions_2.csv"))
    df1 = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    # print(df1)
    ax = sns.lineplot(x='head_x', y='head_y', data=df1, label='Indirect', sort=False)
    ax2 = sns.lineplot(x='head_x', y='head_y', data=df2, label='Direct', sort=False)
    ax3 = sns.lineplot(x="target_x", y="target_y", data=df2, linewidth=2, alpha=0.7, label="Target", sort=False)

    plt.plot(-46.324642181396484, -0.9814049601554871, color='C0', marker='o')  # indirect
    plt.plot(-42.495506286621094, -0.5429293513298035, color='C1', marker='o')  # direct
    plt.plot(-38.11226417004295, -2.997188290909695, color='C2', marker='o')  # target

    # plt.plot(-41.0197639465332, -13.478766441345215, color='C0', marker='o')  # indirect
    # plt.plot(-41.81261444091797, -17.083084106445312, color='C1', marker='o')  # direct
    # plt.plot(-44.25917368138803, -21.484156850185492, color='C2', marker='o')  # target

    plt.plot(-51.42644119262695, -28.565216064453125, color='C0', marker='o')  # indirect
    plt.plot(-50.69612121582031, -29.701793670654297, color='C1', marker='o')  # direct
    plt.plot(-55.05901745163301, -32.76766273203995, color='C2', marker='o')  # target

    ax.axis('equal')
    # ax2.axis('equal')
    # plt.plot(df1)
    # plt.plot(df2)
    ax.set(ylabel='y [m]', xlabel='x [m]')
    ax.legend()
    # plt.ylim(40, 100)
    plt.show()

    fig = ax.get_figure()
    pdfpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                        "plots", "training_data", "circle_traj_plot.pdf"))
    pngpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                        "plots", "training_data", "circle_traj_plot.png"))
    fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')


def plot_traj_line():
    fname = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                      "plots", "line-09-26", "indirect", "positions_8.csv"))
    fname2 = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                       "plots", "line-09-26", "direct", "positions_5.csv"))
    df1 = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)
    print(df1)
    ax = sns.lineplot(x='head_x', y='head_y', data=df1, label='Indirect', sort=False)
    ax2 = sns.lineplot(x='head_x', y='head_y', data=df2, label='Direct', sort=False)
    # ax3 = sns.lineplot(x='target_x', y='target_y', data=df2, label='Target', sort=False)
    # plt.plot(df1)
    # plt.plot(df2)
    target_traj_x = np.arange(-46, -9, 1)
    target_traj_y = np.zeros((37,))
    ax3 = sns.lineplot(target_traj_x, target_traj_y, linewidth=2, alpha=0.7, label='Target')

    plt.plot(-14.92916202545166, -0.6757935285568237, color='C0', marker='o')  # indirect
    plt.plot(-14.953131675720215, -0.0902303159236908, color='C1', marker='o')  # direct
    plt.plot(-9.919937, 0.0, color='C2', marker='o')  # target end
    # plt.plot(target_traj_x, target_traj_y, marker='', color='C2', linewidth=2, alpha=0.6)  # target traj

    ax.set(ylabel='y [m]', xlabel='x [m]')
    ax.axis('equal')

    ax.legend()
    # plt.ylim(-10, 10)
    plt.show()

    fig = ax.get_figure()
    pdfpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                        "plots", "training_data", "line_traj_plot.pdf"))
    pngpath = str(pathlib.Path.joinpath(pathlib.Path.cwd().parent.parent,
                                        "plots", "training_data", "line_traj_plot.png"))
    fig.savefig(pdfpath, dpi=300, bbox_inches='tight')
    fig.savefig(pngpath, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    sns.set()
    sns.set_context('talk')
    # plot_h_fun_csv()
    # var_stddev()
    # plot_bars2()
    # plot_training_reward()
    # plot_training_eplen()
    # plot_ep_this_iter()
    # plot_trajectory_circle()
    # plot_traj_line()
    plot_bars_presentation()
