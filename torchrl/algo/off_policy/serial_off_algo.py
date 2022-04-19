import time
import numpy as np
import math

import torch

from torchrl.algo.rl_algo import RLAlgo

class SerialOffAlgo(RLAlgo):
    """
    Base RL Algorithm Framework, with serialized collector
    """
    def __init__(self,

        pretrain_epochs=0,

        min_pool = 0,

        target_hard_update_period = 1000,
        use_soft_update = True,
        tau = 0.001,
        opt_times = 1,

        **kwargs
    ):
        super(SerialOffAlgo, self).__init__(**kwargs)

        # environment relevant information
        self.pretrain_epochs = pretrain_epochs
        
        # target_network update information
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        # training information
        self.opt_times = opt_times
        self.min_pool = min_pool

        self.sample_key = [ "obs", "next_obs", "acts", "rewards", "terminals" ]

    def update_per_timestep(self):
        if self.replay_buffer.num_steps_can_sample() > max( self.min_pool, self.batch_size ):
            for _ in range( self.opt_times ):
                batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                infos = self.update( batch )
                self.logger.add_update_info( infos )

    def update_per_epoch(self):
        for _ in range( self.opt_times ):
            batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
            infos = self.update( batch )
            self.logger.add_update_info( infos )

    
    def pretrain(self):
        total_frames = 0
        self.pretrain_epochs * self.collector.worker_nums * self.epoch_frames
        
        for pretrain_epoch in range( self.pretrain_epochs ):

            start = time.time()

            self.start_epoch()
            
            # serialized collector requires passing policy network and input dimension every time
            training_epoch_info =  self.collector.train_one_epoch(policy = self.policy)
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            finish_epoch_info = self.finish_epoch()

            total_frames += self.collector.active_worker_nums * self.epoch_frames
            
            infos = {}
            
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(self.training_episode_rewards)
            infos.update(finish_epoch_info)
            
            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos, csv_write=False )
        
        self.pretrain_frames = total_frames
        self.logger.log("Finished Pretrain")


    def train(self):
        self.pretrain()
        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames

        self.start_epoch()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            start = time.time()

            self.start_epoch()
            
            explore_start_time = time.time()
            
            # get rewards from env
            # info = {  'train_rewards' : train_rews = [],
            #           'train_epoch_reward': train_epoch_reward =0 }
            training_epoch_info =  self.collector.train_one_epoch(policy = self.policy)
            print("training epoch info:", training_epoch_info)
            
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)
            explore_time = time.time() - explore_start_time

            train_start_time = time.time()
            self.update_per_epoch()
            train_time = time.time() - train_start_time

            finish_epoch_info = self.finish_epoch()

            eval_start_time = time.time()
            eval_infos = self.collector.eval_one_epoch(policy = self.policy)
            eval_time = time.time() - eval_start_time

            total_frames += self.collector.active_worker_nums * self.epoch_frames

            infos = {}

            for reward in eval_infos["eval_rewards"]:
                self.episode_rewards.append(reward)
            # del eval_infos["eval_rewards"]
            
            cnt = 0
            # if epoch %100 ==0:
            #     for images in eval_infos["image"]:
            #         cnt+= 1
            #         imageio.mimsave(self.save_fig_dir + "/epoch"+str(epoch)+"_trial"+str(cnt)+".gif", images)
            
            # print("env: ", eval_infos["eval_cls"])
            # print(eval_infos["act"])
            # exit(0)
            

            # if self.best_eval is None or \
            #     np.mean(eval_infos["eval_rewards"]) > self.best_eval:
            #     self.best_eval = np.mean(eval_infos["eval_rewards"])
            #     print("best model success rate: ",eval_infos["mean_success_rate"])
            #     self.snapshot(self.save_dir, 'best')
            
            # NEW Record way
            if self.best_success is None or \
                np.mean(eval_infos["mean_success_rate"]) > self.best_success or \
                (np.mean(eval_infos["mean_success_rate"]) == self.best_success and np.mean(eval_infos["eval_rewards"]) > self.best_eval):
                self.best_eval = np.mean(eval_infos["eval_rewards"])
                print("best model success rate: ",eval_infos["mean_success_rate"])
                self.snapshot(self.save_dir, 'best')
            
            if self.best_eval is None or \
                np.mean(eval_infos["eval_rewards"]) > self.best_eval:
                    self.best_eval = np.mean(eval_infos["eval_rewards"])
            
            del eval_infos["eval_rewards"]
            del eval_infos['image']
            del eval_infos["act"]
            del eval_infos["eval_cls"]
            del eval_infos['initial_ob']

            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(
                self.training_episode_rewards)
            infos["Explore_Time"] = explore_time
            infos["Train___Time"] = train_time
            infos["Eval____Time"] = eval_time
            infos.update(eval_infos)
            infos.update(finish_epoch_info)

            self.logger.add_epoch_info(epoch, total_frames,
                time.time() - start, infos )

            if epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, epoch)

        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()