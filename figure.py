import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml


fig_size = 11, 9
color = ['r', 'g', 'b', 'c', 'royalblue', 'darkorange', 'r', 'g', 'b', 'c',]
marker = ['s', 'o']
linewidth = [1, 2, 3, 4]
ms = [10, 20, 15, 20]
font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 23,
         }
font2 = {'family': 'Arial',
         'weight': 'normal',
         'size': 30,
         }
tick_size = 23

yaml_path = 'environment/platform-config.yaml'
with open(yaml_path, 'r') as f:
    cfg = yaml.load(f)
train_config = cfg['train_config']
plot_episodes = train_config['n_episodes']
x = np.linspace(1, plot_episodes, plot_episodes)
x_ticks = np.linspace(0, plot_episodes, 6)
x_label = ['Iteration']
y_label = ['Unit Prices of MUs', 'Amount of Sensing Time (Task 1)', 'Amount of Sensing Time (Task 2)', 'Payoff', 'Payoff', 'Payoff']



def read_data(file_name, obj):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    values = np.array(data[obj]).squeeze()
    if len(values.shape) > 1:
        value = values[1::2, :]
    else:
        value = values[1::2]
    return value


def figure_prices(file, save_p):
    obj = 'prices'
    data = read_data(file, obj)
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(len(data[0])):
        ax.plot(x, data[:, i], label='MU{}'.format(i+1), linewidth=linewidth[1])
    ax.legend(prop=font1, loc='upper right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[0], font2)
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'UnitPrice.png', format='png')
    plt.show()

def figure_allocation(file, save_p):
    obj = 'allocation'
    data = read_data(file, obj)
    ti1 = np.squeeze(data[:, :, 0])
    ti2 = np.squeeze(data[:, :, 1])
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(len(data[0])):
        ax.plot(x, ti1[:, i], label='MU{}'.format(i+1), linewidth=linewidth[1])
    ax.legend(prop=font1, loc='upper right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[1], font2)
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'Amount1.png', format='png')

    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(len(data[0])):
        ax.plot(x, ti2[:, i], label='MU{}'.format(i+1), linewidth=linewidth[1])
    ax.legend(prop=font1, loc='upper right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[2], font2)
    # ax.set(xlim=[0, 1000], ylim=[0, 20])
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'Amount2.png', format='png')
    plt.show()


def figure_mu_payoff(file, save_p):
    obj = 'MU_earnings'
    data = read_data(file, obj)
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(len(data[0])):
        ax.plot(x, data[:, i], label='MU{}'.format(i+1), linewidth=linewidth[1])
    ax.legend(prop=font1, loc='lower right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[3], font2)
    # ax.set(xlim=[0, 1000], ylim=[0, 12])
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'PayoffMU.png', format='png')
    plt.show()

def figure_ti_payoff(file, save_p):
    obj_ti = 'TI_earnings'
    obj_mu = 'MU_earnings'
    data_ti = read_data(file, obj_ti)
    data_mu = read_data(file, obj_mu)
    data_ti_sum = np.sum(data_ti, axis=1)
    data_mu_sum = np.sum(data_mu, axis=1)
    data_sp = data_ti_sum + data_mu_sum
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(len(data_ti[0])):
        ax.plot(x, data_ti[:, i], label='TI{}'.format(i+1), linewidth=linewidth[1])
    ax.plot(x, data_sp, color=color[2], label='SP', linewidth=linewidth[1])
    ax.legend(prop=font1, loc='lower right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[4], font2)
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'PayoffSP.png', format='png')
    plt.show()


def figure_sum_mu_payoff(file, task_budget, save_p):
    obj_mu = 'MU_earnings'
    data_budget = np.ones((plot_episodes, 1))*task_budget*2
    data_mu = read_data(file, obj_mu)
    data_mu_sum = np.sum(data_mu, axis=1)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(x, data_mu_sum, label='Sum Payoff of MUs', linewidth=linewidth[1])
    ax.plot(x, data_budget, color=color[0], linestyle="--", label='Sum Budget of SP', linewidth=linewidth[2])
    ax.legend(prop=font1, loc='lower right')
    ax.set_xlabel(x_label[0], font2)
    ax.set_ylabel(y_label[5], font2)
    # ax.set(xlim=[0, 1000], ylim=[0, 55])
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid()
    plt.savefig(save_p+'SumPayoff.png', format='png')
    plt.show()

    
def Figure(task_budget, time_budget, random_seed):
    mode = 'greedy'
    file_path = '/results/train_data/' + 'task_budget=' + str(task_budget) + '-time_budget=' + str(time_budget) + \
        '-seed=' + str(random_seed) + '-mode=' + mode + '/' + str(plot_episodes) + '.pkl'
    save_path = '/results/figures/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    figure_prices(file_path, save_path)  
    figure_allocation(file_path, save_path)
    figure_mu_payoff(file_path, save_path)
    figure_ti_payoff(file_path, save_path)
    figure_sum_mu_payoff(file_path, task_budget, save_path)

