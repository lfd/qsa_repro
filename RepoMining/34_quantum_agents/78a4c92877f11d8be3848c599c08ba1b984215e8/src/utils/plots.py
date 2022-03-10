
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os

from tensorflow.python.keras.models import load_model

from config import BASE_PATH


def plot_avg_score():
    ...


def plot_val_by_dir(val):
    path = '../../../../' + BASE_PATH + 'cartpole/fixed_range_90/'

    for file_name in os.listdir(path):
        if file_name[-13:] == 'scores.pickle':
            try:
                with open(path + file_name.replace('scores', 'meta'), 'rb') as file:
                    meta = pickle.load(file)
                    pprint(meta)

                learning_rate = meta.get('learning_rate')
                update_after = meta.get('update_after')
                update_target_after = meta.get('update_target_after')
                l1u = meta.get('l1_units')
                l2u = meta.get('l2_units')

                if True:
                    with open(path + file_name.replace('scores', val), 'rb') as file:
                        data = pickle.load(file)
                        # print(file_name)
                        # print(data)

                        plt.plot(data)
                        plt.title("Solved at episode {}".format(str(meta.get('env_solved_at')).replace('[]', '(not solved)')))
                        plt.xlabel('Episode')
                        plt.ylabel('Score')
                        plt.show()

            except Exception as e:
                print("Error with model", file_name)
                print(e)


def plot_avg_vals(val, min_val, avg_over, path, label, color, hyperparams, plot_to=None, plt_obj=None, avg_solved=False):
    all_vals = []
    env_solved = []

    best_agent_solved_at = 10000000
    best_agent_name = 'not found'
    best_agent_meta = 'not found'

    for file_name in os.listdir(path):
        if file_name[-13:] == 'scores.pickle' and file_name[:5] != 'dummy':
            try:
                with open(path + file_name.replace('scores', 'meta'), 'rb') as file:
                    meta = pickle.load(file)
                    # print(meta)

                include_agent = True
                for hp, value in hyperparams.items():
                    if meta.get(hp) != value:
                        include_agent = False
                        break
                    else:
                        if meta['env_solved_at']:
                            env_solved.append(meta['env_solved_at'][0])

                if include_agent:
                    # print(meta)
                    with open(path + file_name.replace('scores', val), 'rb') as file:
                        data = pickle.load(file)
                        # pprint(meta['env_solved_at'])
                        # print(file_name)

                        if isinstance(data[0], list):
                            concatenated = []
                            for element in data:
                                concatenated += element
                            data = concatenated

                        # plt.plot(data)
                        # plt.show()

                        filled_vals = np.ones(shape=min_val) * data[-1]
                        filled_vals[:len(data)] = data

                        if meta['env_solved_at'] and meta['env_solved_at'][0] < best_agent_solved_at:
                            best_agent_solved_at = meta['env_solved_at'][0]
                            best_agent_name = file_name
                            best_agent_meta = meta

                        # plt.plot(filled_vals)
                        # plt.show()

                        if len(all_vals) < avg_over:
                            all_vals.append(filled_vals)
                        else:
                            break
            except Exception as e:
                print("Error in file", file_name)
                print(e)

    print(len(all_vals))
    if len(all_vals) == 0:
        pprint(hyperparams)

    print("best agent:", best_agent_solved_at)
    print("best agent name:", best_agent_name)
    print("best agent meta:", best_agent_meta)

    clipped_vals = [x[:min_val] for x in all_vals]
    all_vals = clipped_vals
    mean_vals = np.mean(all_vals, axis=0)
    error = np.std(all_vals, axis=0)
    # error = sem(all_vals)

    fill_low = np.clip(np.asarray(mean_vals) - np.asarray(error), 0, None)
    fill_high = np.clip(np.asarray(mean_vals) + np.asarray(error), None, 200)

    # sb.set_style("whitegrid")

    if plt_obj:
        if plot_to is not None:
            plt_obj.plot(list(range(plot_to)), mean_vals[:plot_to], color=color, label=label)
            plt_obj.fill_between(range(plot_to), fill_low[:plot_to], fill_high[:plot_to], color=color, lw=0, alpha=0.3)
        else:
            plt_obj.plt_obj(list(range(len(mean_vals))), mean_vals, color=color, label=label)
            plt_obj.fill_between(range(len(error)), fill_low, fill_high, color=color, lw=0, alpha=0.3)
    else:
        if plot_to is not None:
            sb.lineplot(list(range(plot_to)), mean_vals[:plot_to], color=color, label=label)
            plt.fill_between(range(plot_to), fill_low[:plot_to], fill_high[:plot_to], color=color, lw=0, alpha=0.3)
        else:
            sb.lineplot(list(range(len(mean_vals))), mean_vals, color=color, label=label)
            plt.fill_between(range(len(error)), fill_low, fill_high, color=color, lw=0, alpha=0.3)

    if avg_solved:
        avg_env_solved = np.mean(env_solved)
        plt_obj.vlines(avg_env_solved, 0, 200, colors='grey', linestyles='--', )

    # plt.xlabel("Episode")
    # plt.ylabel("Score")
    # plt.ylim(ymax=200)
    # plt.show()


def plot_avg_score():
    path = '../../../../' + BASE_PATH + 'cartpole/hp_search/'

    for file_name in os.listdir(path):
        if file_name[-13:] == 'losses.pickle':
            with open(path + file_name.replace('losses', 'meta'), 'rb') as file:
                meta = pickle.load(file)
                pprint(meta)

            learning_rate = meta.get('learning_rate')
            update_after = meta.get('update_after')
            update_target_after = meta.get('update_target_after')
            epsilon_schedule = meta.get('epsilon_schedule')
            memory_length = meta.get('memory_length')
            batch_size = meta.get('batch_size')

            with open(path + file_name.replace('losses', 'scores'), 'rb') as file:
                data = pickle.load(file)
                # print(file_name)
                # print(data)

                average_scores = []
                window_size = 100
                num_wins = int(len(data)/window_size)
                for i in range(len(data)-window_size):
                    win_score = np.mean(data[i:i+window_size])
                    average_scores.append(win_score)

                plt.plot(average_scores)
                plt.ylim(ymax=200)
                plt.ylabel('Average score')
                plt.xlabel('Episode')
                plt.show()


def plot_by_model_name(model_name, val, path, label, color):
    with open(path + model_name + '_' + val + '.pickle', 'rb') as file:
        data = pickle.load(file)
        print(data)

        plt.plot(data, label=label, color=color)
        plt.title("Final {}: {}".format(val, data[-1]))
        plt.show()
