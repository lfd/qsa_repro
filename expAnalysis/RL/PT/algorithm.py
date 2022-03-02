import numpy as np
from datetime import datetime
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from PT.replay import ReplayMemory, Sampler
from PT.policies import ActionValuePolicy, EpsilonGreedyPolicy, LinearDecay

class Algorithm:

    def __init__(self,  env, val_env, 
                        policy_model, target_model, 
                        replay_capacity, 
                        epsilon_duration, epsilon_start, epsilon_end, 
                        gamma, optimizer_vqc, optimizer_cl, loss,
                        num_steps,
                        update_every,
                        train_after = 1000,
                        train_every = 1,
                        batch_size = 32,
                        validate_every = 100,
                        num_val_trials = 1,
                        name='run',
                        device=torch.device("cuda")):

        self.env = env
        self.val_env = val_env

        self.policy_model = policy_model
        self.target_model= target_model
        
        self.greedy_policy = ActionValuePolicy(policy_model)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.greedy_policy, self.env.action_space
        )

        self.memory = ReplayMemory(replay_capacity, device)

        self.epsilon_schedule = LinearDecay(
            policy=self.behavior_policy,
            num_steps=epsilon_duration,
            start=epsilon_start,
            end=epsilon_end
        )

        self.num_steps = num_steps
        self.update_every = update_every
        self.train_after = train_after
        self.train_every = train_every
        self.batch_size = batch_size
        self.validate_every = validate_every
        self.gamma = gamma
        self.optimizer_vqc = optimizer_vqc
        self.optimizer_cl = optimizer_cl
        self.loss = loss
        self.num_val_trials = num_val_trials
        self.val_step = 0
        self.device=device

        experiment = f'logs/{name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.summary_writer = SummaryWriter(log_dir=experiment)



    def train(self):

        sampler = Sampler(self.behavior_policy, self.env)
        
        for step in range(self.num_steps):

            train_step = step - self.train_after
            is_training = train_step >= 0

            if is_training:
                self.epsilon_schedule.step()

            transition = sampler.step()
            self.memory.store(transition)

            if not is_training:
                continue

            if train_step % self.train_every == 0:
                
                with torch.no_grad():
                    # Sample a batch of transitions from the replay buffer
                    batch = self.memory.sample(self.batch_size)
                    
                    # Check whether the batch contains next_states (the sampled
                    # batch might contain terminal states only)
                    if len(batch.next_states) > 0:
                        target_next_q_values = self.target_model(batch.next_states)

                        target_next_v_values = torch.max(
                            target_next_q_values, 
                            dim=-1
                        ).values

                        non_terminal_indices = torch.where(~batch.is_terminal)[0]

                        targets = torch.scatter_add(
                            batch.rewards,
                            index=non_terminal_indices,
                            src=target_next_v_values,
                            dim=0
                        )                

                self.optimizer_vqc.zero_grad()
                self.optimizer_cl.zero_grad()

                policy_q_values = self.policy_model(batch.states)

                action_indices = torch.unsqueeze(batch.actions, dim=-1)

                policy_v_values = torch.gather(
                    policy_q_values, 
                    index=action_indices, 
                    dim=1
                )

                policy_v_values = torch.squeeze(
                    policy_v_values,
                    axis=-1
                )

                loss = self.loss(targets, policy_v_values)

                loss.backward()

                self.optimizer_vqc.step()

                if self.optimizer_cl:
                    self.optimizer_cl.step()

            if train_step % self.update_every == 0:
                self.target_model = copy.deepcopy(self.policy_model)
                self.target_model.to(self.device)

            if train_step % self.validate_every == 0:
                self.validate()

        self.summary_writer.close()


    def validate(self):
        returns = []
        total_reward=0
        obs, done = self.val_env.reset(), False

        for _ in range (self.num_val_trials):
            
            while not done:
                action = self.greedy_policy(torch.as_tensor(obs).float())

                obs, reward, done, _ = self.val_env.step(action)
                total_reward += reward

            returns.append(total_reward)
            total_reward = 0
            obs, done = self.val_env.reset(), False

        val_return = np.mean(returns)
        
        self.summary_writer.add_scalar('val_step/avg_return', val_return, self.val_step)

        self.val_step += 1

