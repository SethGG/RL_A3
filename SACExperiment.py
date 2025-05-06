import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from SACAgent import SACAgent
import os
import torch.multiprocessing as mp


def evaluation(agent: SACAgent):
    # Evaluate the agent's performance in the environment
    env = gym.make('CartPole-v1')
    s, info = env.reset()
    done = False
    trunc = False
    episode_return = 0
    while not done and not trunc:
        a = agent.select_action_greedy(s)
        s_next, r, done, trunc, info = env.step(a)
        episode_return += r
        s = s_next
    return episode_return


def run_single_repetition(task):
    # Run a single repetition of the experiment
    config_id, rep_id, n_envsteps, eval_interval, params = task
    params = params.copy()

    update_freq = params.pop("update_freq")

    # Create a new environment and agent for each repetition
    env = gym.make('CartPole-v1')
    params["state_dim"] = env.observation_space.shape[0]
    params["n_actions"] = env.action_space.n

    agent = SACAgent(**params)

    eval_returns = np.zeros(int(n_envsteps / eval_interval))
    eval_num = 0

    s, info = env.reset()
    for envstep in range(1, n_envsteps+1):
        a = agent.select_action_sample(s)
        s_next, r, done, trunc, info = env.step(a)
        agent.add_experience(s, a, r, s_next, done)

        if envstep % update_freq == 0:
            for _ in range(update_freq):
                agent.update()

        s = s_next

        if done or trunc:
            s, info = env.reset()

        if envstep % eval_interval == 0:
            # Evaluate the agent periodically
            eval_return = evaluation(agent)
            eval_returns[eval_num] = eval_return
            eval_num += 1
            print(f"Running config: {config_id+1:2}, Repetition {rep_id+1:2}, Environment steps: {envstep:6}, "
                  f"Eval return: {eval_return:3}")

    return config_id, eval_returns


def conf_filename(outdir, params, suffix):
    # Generate a filename for saving results based on parameters
    filename = "_".join(f"{key}_{value}" for key, value in params.items()) + f"_{suffix}.csv"
    return os.path.join(outdir, filename)


def run_experiments(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval):
    # Run experiments with different parameter combinations
    processes = 3  # Number of parallel processes

    os.makedirs(outdir, exist_ok=True)

    tasks = []
    for config_id, params in enumerate(param_combinations):
        if params in (t[-1] for t in tasks):
            print(f"Configuration {config_id+1} is already present in the task list. Skipping...")
            continue
        if os.path.exists(conf_filename(outdir, params, "eval")):
            print(f"Results for configuration {config_id+1} already exist. Skipping...")
            continue
        for rep_id in range(n_repetitions):
            tasks.append((config_id, rep_id, n_envsteps, eval_interval, params))

    results_by_config = {}

    with mp.Pool(processes=processes) as pool:
        for config_id, result_eval in pool.imap(run_single_repetition, tasks):
            if config_id not in results_by_config:
                results_by_config[config_id] = []
            results_by_config[config_id].append(result_eval)

            if len(results_by_config[config_id]) == n_repetitions:
                results_eval = np.array(results_by_config[config_id])
                np.savetxt(conf_filename(outdir, param_combinations[config_id], "eval"), results_eval, delimiter=",")


def create_plot(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval, title, label_params, plotfile):
    # Create plots for the experiment results
    smoothing_window = 3
    plot = LearningCurvePlot(title)

    for params in param_combinations:
        results_eval = np.loadtxt(conf_filename(outdir, params, "eval"), delimiter=",", ndmin=2)
        mean_results_eval = np.mean(results_eval, axis=0)
        conf_results_eval = np.std(results_eval, axis=0) / np.sqrt(n_repetitions)

        plot.add_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), smooth(mean_results_eval,
                       window=smoothing_window), smooth(conf_results_eval, window=smoothing_window),
                       label=", ".join(f"{p}: {params[p]}" for p in label_params if p in params))

    plot.save(name=plotfile)


if __name__ == '__main__':
    param_combinations = [
        {"lr": 0.001, "gamma": 0.99, "hidden_dim": 64, "alpha": 0.2, "buffer_size": 10000, "batch_size": 100,
         "learning_starts": 1000, "tau": 0.005, "full_expectation": True, "double_q": True, "update_freq": 50},
        {"lr": 0.001, "gamma": 0.99, "hidden_dim": 64, "alpha": 0.2, "buffer_size": 10000, "batch_size": 100,
         "learning_starts": 1000, "tau": 0.005, "full_expectation": False, "double_q": True, "update_freq": 50}
    ]

    n_repetitions = 3  # Number of repetitions for each experiment
    n_envsteps = 100000  # Number of environment steps
    eval_interval = 1000  # Interval for evaluation
    outdir = f"evaluations_{n_envsteps}_envsteps"  # Output directory for results

    run_experiments(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval)
    create_plot(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval, "Test", ["full_expectation"],
                "test.png")
