import json
import sys
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE
# import sys
sys.path.append(".") 

import torch

import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo.off_policy.mt_sac import SerialMTSAC
from torchrl.collector.para.serial_async_mt import AsyncMultiTaskSerialCollectorUniform
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
import gym

from metaworld_utils.meta_env import get_meta_env, generate_single_mt_env
import random

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    # print(args)
    
    if args.task_env == "MT10_task_env":
        cls_dicts = {args.task_name: EASY_MODE_CLS_DICT[args.task_name]}
        cls_args = {args.task_name: EASY_MODE_ARGS_KWARGS[args.task_name]}
        env_name =  EASY_MODE_CLS_DICT[args.task_name]
        

    elif args.task_env == "MT50_task_env":
        if args.task_name in HARD_MODE_CLS_DICT['train'].keys():     # 45 tasks
            cls_dicts = {args.task_name: HARD_MODE_CLS_DICT['train'][args.task_name]}
            cls_args = {args.task_name: HARD_MODE_ARGS_KWARGS['train'][args.task_name]}
            env_name =  HARD_MODE_CLS_DICT['train'][args.task_name]
        else:
            cls_dicts = {args.task_name: HARD_MODE_CLS_DICT['test'][args.task_name]}
            cls_args = {args.task_name: HARD_MODE_ARGS_KWARGS['test'][args.task_name]}
            env_name =  HARD_MODE_CLS_DICT['test'][args.task_name]

    else:
        raise NotImplementedError
    
    
    env_name =  EASY_MODE_CLS_DICT[args.task_name]
    
    # set cls_args['kwargs']['obs_type'] = params['meta_env']['obs_type']
    cls_args[args.task_name]['kwargs']['obs_type'] = params['meta_env']['obs_type']
    cls_args[args.task_name]['kwargs']['random_init'] = params['meta_env']['random_init']
    
    from metaworld_utils.concurrent_sawyer import ConcurrentSawyerEnv
    cls_dicts = {"push_pick_place": ConcurrentSawyerEnv}
    cls_args = {"push_pick_place": dict(args=[], kwargs={'obs_type': params['meta_env']['obs_type'], 'task_type': ['push', 'pick_place']})}
    env_name =  ConcurrentSawyerEnv
    
    #env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
    env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False)

    
    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0]
    }
    print("example obs: ", example_ob)
    
    # env.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , None, args.seed, params, args.log_dir, args.task_name)

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    

    print("num tasks: ", env.num_tasks)
    pf = policies.MultiHeadGuassianContPolicy(
        input_shape = example_ob.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        head_num=env.num_tasks,
        **params['net'] )

    print(pf)
    
    # pf.load_state_dict(torch.load("../Behavior-Cloning/policy/expert/push-v1.pth", map_location='cpu'))
    # pf.load_state_dict(torch.load("log/MT50_Single_Task/SawyerSweepIntoGoalEnv/sweep-into-v1/32/model/model_pf_best.pth", map_location='cpu'))

    # print("pf: ", pf)
    qf1 = networks.FlattenBootstrappedNet( 
        input_shape = example_ob.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        head_num=env.num_tasks,
        **params['net'] )

    # print("qf1: ", qf1)
    qf2 = networks.FlattenBootstrappedNet( 
        input_shape = example_ob.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        head_num=env.num_tasks,
        **params['net'] )
    print("qf2: ", qf2)
    
    
 
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    # params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
    #     env=env, pf=pf, replay_buffer=replay_buffer,
    #     env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
    #     device=device,
    #     reset_idx=True,
    #     epoch_frames=params['general_setting']['epoch_frames'],
    #     max_episode_frames=params['general_setting']['max_episode_frames'],
    #     eval_episodes = params['general_setting']['eval_episodes'],
    #     worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
    #     train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs'],
    #     eval_render = params['general_setting']['eval_render']
    # )
    
    print("device: ", device)
    params['general_setting']['collector'] = AsyncMultiTaskSerialCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, 
        # eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs'],
        eval_render = params['general_setting']['eval_render'],
        input_shape = example_ob.shape[0]
    )

    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    params['general_setting']['save_fig_dir'] = osp.join(logger.work_dir,"fig")
    
    
    #----------------debug------------------------
    # single_mt_env_args = {
    #         "task_cls": None,
    #         "task_args": None,
    #         "env_rank": 0,
    #         "num_tasks": 1,
    #         "max_obs_dim": np.prod(env.observation_space.shape),
    #         "env_params": params["env"],
    #         "meta_env_params": params["meta_env"]
    #     }
    
    # import copy
    # env_args = single_mt_env_args
    # env_args["task_cls"] = cls_dicts[args.task_name]
    # env_args["task_args"] = copy.deepcopy(cls_args[args.task_name])
    # env_args["env_rank"] = 0
    
    # env = generate_single_mt_env(**env_args)
    # print(env_args)
    # print(env)
    # env.eval()

    # #-----------------evaluating loaded policy network-----------------
    # return_dic = params['general_setting']['collector'].eval_one_epoch()
    # print(return_dic)
    
    # agent = MTMHSAC(
    agent = SerialMTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()
    

if __name__ == "__main__":

    experiment(args)
    
