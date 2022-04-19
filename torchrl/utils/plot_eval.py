import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import argparse
import seaborn as sns
import csv
import json


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, nargs='+', default=(0,),
                        help='random seed (default: (0,))')
    parser.add_argument('--max_m', type=int, default=None,
                        help='maximum million')
    parser.add_argument('--smooth_coeff', type=int, default=25,
                        help='smooth coeff')
    parser.add_argument('--env_name', type=str, default='mt10',
                        help='environment trained on (default: mt10)')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')
    parser.add_argument( "--id", type=str, nargs='+', default=('origin',),
                        help="id for tensorboard")
    parser.add_argument( "--tags", type=str, nargs='+', default=None,
                        help="id for tensorboard")
    parser.add_argument('--output_dir', type=str, default='./fig',
                        help='directory for plot output (default: ./fig)')
    parser.add_argument('--entry', type=str, default='Running_Average_Rewards',
                        help='Record Entry')
    parser.add_argument('--add_tag', type=str, default='',
                        help='added tag')
    parser.add_argument("--task_name", type=str, default=None,
                        help="the task name for single task training")
    args = parser.parse_args()
    return args


args = get_args()
env_name = args.env_name
env_id = args.id
task_name = args.task_name

if args.tags is None:
    args.tags = args.id
assert len(args.tags) == len(args.id)


def post_process(array):
    smoth_para = args.smooth_coeff
    new_array = []
    for i in range(len(array)):
        if i < len(array) - smoth_para:
            new_array.append(np.mean(array[i:i+smoth_para]))
        else:
            new_array.append(np.mean(array[i:None]))
    return new_array    


sns.set("paper")
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.90,
#                 wspace=1, hspace=0)

max_scores = {}

for task_id in args.id:
    
    if task_id == "MT10_Single_Task":
        with open("metaworld_utils/MT10_task_env.json", "r") as f:
            task_env = json.load(f)
    elif task_id == "MT50_Single_Task":
        with open("metaworld_utils/MT50_task_env.json", "r") as f:
            task_env = json.load(f)
    else:
        raise NotImplementedError
        
    for task in task_env.keys():
        max_scores[task] = 0

    for task in task_env.keys():
        for seed in args.seed:
            file_path = os.path.join(args.log_dir, task_id, task_env[task], task, str(seed), 'log.csv')
            with open(file_path,'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    if float(row[args.entry]) > max_scores[task]:
                        max_scores[task] = float(row[args.entry])

    
    y = list(max_scores.keys())
    x = list(max_scores.values())

    sns.set_color_codes("pastel")
    sns.barplot(x=x, y=y, label="Best", color="b", orient="h")


    ax.set(xlim=(0, 1), ylabel="task",
        xlabel="mean-success-rate")
    sns.despine(left=True, bottom=True)

    plt.title("{} mean success rate".format(task_id), fontsize=20)

    if not os.path.exists( args.output_dir ):
        os.mkdir( args.output_dir )

    plt.savefig( os.path.join( args.output_dir, '{}_mean_success_rate.png'.format(task_id) ) )
    plt.close()