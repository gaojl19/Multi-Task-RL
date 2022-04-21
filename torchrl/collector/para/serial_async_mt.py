import torch
import copy
import numpy as np

from .base import AsyncParallelCollector
import torch.multiprocessing as mp

import torchrl.policies as policies

from torchrl.env.get_env import *
from torchrl.env.continuous_wrapper import *
from torchrl.collector.base import EnvInfo

from metaworld_utils.meta_env import generate_single_mt_env

from metaworld_utils.meta_env import get_meta_env

from collections import OrderedDict, deque


class AsyncMultiTaskSerialCollectorUniform():
    def __init__(self, env, pf, replay_buffer,env_cls, env_args,
        train_epochs, eval_epochs, worker_nums, input_shape,
        train_render=False, eval_render=False, eval_episodes=1, epoch_frames=1000,
        device="cpu", max_episode_frames = 999, reset_idx=False):

        self.env = env
        print("training env: ", env)
        self.pf = pf
        self.env.train()
        continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.train_render = train_render
        self.worker_nums = worker_nums
        
        self.eval_env = copy.copy(env)
        self.eval_env._reward_scale = 1
        self.eval_episodes = eval_episodes
        self.eval_render = eval_render

        self.env_info = EnvInfo(
            env, device, train_render, eval_render,
            epoch_frames, eval_episodes,
            max_episode_frames, continuous, None
        )
        self.c_ob = {
            "ob": self.env.reset()
        }

        self.train_rew = 0
        self.training_episode_rewards = deque(maxlen=20)
        
        # device specification
        self.device = device
        # print("collector device: ", device)

        self.to(self.device)

        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        self.env_cls  = env_cls
        self.env_args = env_args
        self.replay_buffer = replay_buffer
        self.train_epochs = train_epochs
        self.eval_epochs = eval_epochs
        
        # # cpu for multiprocess sampling
        # self.env_info.device = 'cpu'
        
        self.reset_idx = reset_idx
        
        self.tasks = list(self.env_cls.keys())
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        
        self.shared_dict = {}
        self.input_shape = input_shape
    
    def to(self, device):
        for func in self.funcs:
            self.funcs[func].to(device)
    
    @property
    def funcs(self):
        return {
            "pf": self.pf
        }
    
    @classmethod
    def take_actions(cls, pf, env_info, ob_info, replay_buffer, input_shape):
        
        ob = ob_info["ob"]
        # Adjust ob shape
        ob = ob[:input_shape]
        
        task_idx = env_info.env_rank
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)

        embedding_flag = isinstance(pf, policies.EmbeddingGuassianContPolicyBase)

        pf.eval()

        with torch.no_grad():
            if idx_flag:
                idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                if embedding_flag:
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), embedding_input,
                        [task_idx])
                else:
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
                        idx_input)
                act = out["action"]
                # act = act[0]
            else:
                if embedding_flag:
                    # embedding_input = np.zeros(env_info.num_tasks)
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), embedding_input)
                else:    
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0))
                act = out["action"]


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = env_info.env.step(act)
        
        # Adjust next ob
        next_ob = next_ob[:input_shape]
        
        ############### check step reward  ################
        # print("step reward: ", reward)  -0.17927 dense reward
        
        
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.env_rank],
            "rewards": [reward],
            "terminals": [done]
        }
        if embedding_flag:
            sample_dict["embedding_inputs"] = embedding_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step
        
        replay_buffer.add_sample(sample_dict, env_info.env_rank)

        return next_ob, done, reward, info
    
    
    def train_one_epoch(self, policy):
        
        tasks = list(self.env_cls.keys())
        # self.shared_dict = self.manager.dict()
                
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }
        
        train_rews = []
        goal_dist = []
        return_dist = []
        train_epoch_reward = 0
        active_worker_nums = 0

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            result = self.train(policy=policy,
                          env_info=self.env_info,
                          task_name=task)
                        #   shared_dict=self.shared_dict)
            train_rews += result["train_rewards"]
            train_epoch_reward += result["train_epoch_reward"]
            goal_dist += result["goal_dist"]
            return_dist += result["return_dist"]
            active_worker_nums += 1
        
        self.active_worker_nums = active_worker_nums
        
        
        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward,
            'goal_dist': goal_dist,
            'return_dist': return_dist
        }
        
        
    def eval_one_epoch(self, policy, render=False):
        
        tasks = list(self.env_cls.keys())
                
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }
        
        eval_rews = []
        images = []
        actions = []
        initial_obs = []
        
        mean_success_rate = 0
        tasks_result = []
        active_task_counts = 0  
               
        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            result = self.evaluate(policy=policy,
                          env_info=self.env_info,
                          eval_episode=self.env_info.eval_episodes,
                          max_frame=self.env_info.max_episode_frames,
                          task_name=task,
                          render=render)
            
            active_task_counts += 1
            eval_rews += result["eval_rewards"]
            images += result["image_obs"]
            mean_success_rate += result["success_rate"]
            tasks_result.append((result["task_name"], result["success_rate"], np.mean(result["eval_rewards"])))
        
        
        tasks_result.sort()
    
        dic = OrderedDict()
        for task_name, success_rate, eval_rewards in tasks_result:
            dic[task_name+"_success_rate"] = success_rate
            dic[task_name+"_eval_rewards"] = eval_rewards
            
            # if self.tasks_progress[self.tasks_mapping[task_name]] is None:
            #     self.tasks_progress[self.tasks_mapping[task_name]] = success_rate
            # else:
            # self.tasks_progress[self.tasks_mapping[task_name]] *= \
            #     (1 - self.progress_alpha)
            # self.tasks_progress[self.tasks_mapping[task_name]] += \
            #     self.progress_alpha * success_rate

        dic['eval_rewards']      = eval_rews
        dic['image']             = images       #[[trial1],[trial2]...]
        dic['mean_success_rate'] = mean_success_rate / active_task_counts
        dic['act']               = actions
        dic['eval_cls']          = env_cls
        dic['initial_ob']        = initial_obs

        return dic
    
    
    def train(self, policy, env_info, task_name):
        
        pf = copy.deepcopy(policy)
       
        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)
        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        env_info.env.eval()
        env_info.env._reward_scale = 1
    
        # initialize
        done = False
        train_rews = []
        goal_dist = []
        return_dist = []
        train_rew = 0
        train_epoch_reward = 0
        
        c_ob = {
            "ob": env_info.env.reset()
        }
        
        if norm_obs_flag:
            self.shared_dict[task_name] = {
                "obs_mean": env_info.env._obs_mean,
                "obs_var": env_info.env._obs_var
            }    

        for _ in range(env_info.epoch_frames):
            # print(env_info.epoch_frames)
            next_ob, done, reward, _ = self.take_actions(pf, env_info, c_ob, self.replay_buffer, self.input_shape)
            c_ob["ob"] = next_ob
            train_rew += reward
            train_epoch_reward += reward
            if done:
                train_rews.append(train_rew)
                goal_dist.append(_["goalDist"])
                return_dist.append(_["returnDist"])
                train_rew = 0

            if norm_obs_flag:
                self.shared_dict[task_name] = {
                    "obs_mean": env_info.env._obs_mean,
                    "obs_var": env_info.env._obs_var
                }
                # print("Put", task_name)

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward,
            'goal_dist': goal_dist,
            'return_dist': return_dist
        }
            
    
    def evaluate(self, policy, env_info, eval_episode, max_frame, task_name, render):

        pf = copy.deepcopy(policy)
        pf.eval()
        
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.EmbeddingGuassianContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)
        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        env_info.env.eval()
        env_info.env._reward_scale = 1
        round = 0
        success = 0
        rew = 0
        image_obs = []
        
        # print("max episode frames: ", env_info.max_episode_frames)
        for i in range(eval_episode):
            if norm_obs_flag:
                env_info.env._obs_mean = self.shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = self.shared_dict[task_name]["obs_var"]
            
            # initialize
            acs = []
            done = False
            eval_rews = []
        
            eval_ob = env_info.env.reset()
            task_idx = env_info.env_rank
            current_success = 0
            current_step = 0
            
            while not done:
                eval_ob = eval_ob[:self.input_shape]
                if idx_flag:
                    idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),
                            embedding_input, [task_idx] )
                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    
                else:
                    if embedding_flag:
                        # create embedding
                        embedding_input = torch.zeros(env_info.num_tasks)
                        embedding_input[env_info.env_rank] = 1
                        embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                        
                        # mask out the last 3 dimensions
                        # eval_ob = eval_ob[:9]

                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input)
                    else:
                        act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                
                acs.append(act)

                eval_ob, r, done, info = env_info.env.step( act )
                rew += r
                current_success = max(current_success, info["success"])
                current_step += 1
                
                # render
                if render==True:
                    if i == int(eval_episode/2):
                        image = env_info.env.get_image(400,400,"leftview")
                        # print(env_info.env.get_image(400,400,"frontview"))
                        image_obs.append(image)
                    
                done = False
                # here we should wait until tasks are all done
                if current_step > max_frame:
                    next_ob = env_info.env.reset()
                    env_info.finish_episode()
                    break
                
            eval_rews.append(rew)
            success += current_success
            round += 1

        success_rate = success / round
        
        return { 
            'eval_rewards': eval_rews,
            'image_obs': image_obs, 
            'success_rate': success_rate, 
            'task_name': task_name
        }