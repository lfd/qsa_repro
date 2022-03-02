import tensorflow as tf
import numpy as np
from datetime import datetime

from TF.replay import ReplayMemory, Sampler
from TF.policies import ActionValuePolicy, EpsilonGreedyPolicy, LinearDecay

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
                        name='run'):

        self.env = env
        self.val_env = val_env

        self.policy_model = policy_model
        self.target_model= target_model
        
        self.greedy_policy = ActionValuePolicy(policy_model)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.greedy_policy, self.env.action_space
        )

        self.memory = ReplayMemory(replay_capacity)

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

        experiment = f'logs/{name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.summary_writer = tf.summary.create_file_writer(experiment)



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
                
                # Sample a batch of transitions from the replay buffer
                batch = self.memory.sample(self.batch_size)

                # Convert target to correct datatype (PennyLane requires 64bit
                # floats, Cirq/TFQ works with standard 32bit)
                targets = tf.cast(batch.rewards, tf.keras.backend.floatx())

                # Check whether the batch contains next_states (the sampled
                # batch might contain terminal states only)
                if len(batch.next_states) > 0:
                    target_next_q_values = self.target_model(batch.next_states)
                    target_next_v_values = tf.reduce_max(
                        target_next_q_values, 
                        axis=-1
                    )
                    
                    non_terminal_indices = tf.where(~batch.is_terminal)
                    
                    targets = tf.cast(batch.rewards, target_next_v_values.dtype)

                    targets = tf.tensor_scatter_nd_add(
                        targets,
                        non_terminal_indices,
                        self.gamma * target_next_v_values
                    )

                with tf.GradientTape() as tape:
                    policy_q_values = self.policy_model(batch.states)

                    action_indices = tf.expand_dims(batch.actions, axis=-1)

                    policy_v_values = tf.gather(
                        policy_q_values, 
                        action_indices, 
                        batch_dims=1
                    )

                    policy_v_values = tf.squeeze(
                        policy_v_values,
                        axis=-1
                    )

                    loss = self.loss(targets, policy_v_values)

                grads = tape.gradient(
                    loss, 
                    self.policy_model.trainable_variables
                )

                self.optimizer_vqc.apply_gradients(
                        zip([grads[0]], [self.policy_model.trainable_variables[0]])
                    )
                    
                self.optimizer_cl.apply_gradients(
                        zip([grads[1]], [self.policy_model.trainable_variables[1]])
                    )

            if train_step % self.update_every == 0:
                self.target_model.set_weights(
                    self.policy_model.get_weights()
                )

            if train_step % self.validate_every == 0:
                self.validate()


    def validate(self):
        returns = []
        total_reward=0
        obs, done = self.val_env.reset(), False

        for _ in range (self.num_val_trials):
            
            while not done:
                action = self.greedy_policy(obs)

                obs, reward, done, _ = self.val_env.step(action)
                total_reward += reward

            returns.append(total_reward)
            total_reward = 0
            obs, done = self.val_env.reset(), False

        val_return = np.mean(returns)
        
        with self.summary_writer.as_default():
            tf.summary.scalar('val_step/avg_return', val_return, self.val_step)

        self.val_step += 1

