import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from DQNAgent import DQNAgent
import os
import torch.multiprocessing as mp


def evaluation(agent: DQNAgent):
    # Evaluate the agent's performance in the environment
    env = gym.make('CartPole-v1')
    s, info = env.reset()
    done = False
    trunc = False
    episode_return = 0
    while not done and not trunc:
        a = agent.select_action(s, -1)  # greedy evaluation
        s_next, r, done, trunc, info = env.step(a)
        episode_return += r
        s = s_next
    return episode_return


def run_single_repetition(task):
    # Run a single repetition of the experiment
    config_id, rep_id, n_envsteps, eval_internal, params = task
    alpha = params["alpha"]
    gamma = params["gamma"]
    update_freq = params["update_freq"]
    epsilon = params["epsilon"]
    decay_rate = params["decay_rate"]
    hidden_dim = params["hidden_dim"]
    tn = params["tn"]
    er = params["er"]

    tn_update_freq = 5  # Frequency of target network updates

    # Create a new environment and agent for each repetition
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    eval_returns = np.zeros(int(n_envsteps / eval_internal))
    eval_epsilon = np.zeros(int(n_envsteps / eval_internal))
    agent = DQNAgent(n_actions, n_states, alpha, gamma, update_freq, hidden_dim, tn, er)

    eval_num = 0
    tn_update_count = 0
    s, info = env.reset()
    for step in range(1, n_envsteps+1):
        a = agent.select_action(s, epsilon)
        s_next, r, done, trunc, info = env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next

        if done or trunc:
            # Reset the environment and decay epsilon
            s, info = env.reset()
            epsilon *= decay_rate
            tn_update_count += 1
            if tn_update_count == tn_update_freq:
                agent.update_tn()
                tn_update_count = 0

        if step % eval_internal == 0:
            # Evaluate the agent periodically
            eval_return = evaluation(agent)
            eval_returns[eval_num] = eval_return
            eval_epsilon[eval_num] = epsilon
            eval_num += 1

            print(f"Running config: {config_id+1:2}, Repetition {rep_id+1:2}, Environment steps: {step:6}, "
                  f"Epsilon: {epsilon:7}, Eval return: {eval_return:3}")

    return config_id, eval_returns, eval_epsilon


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
        for config_id, result_eval, result_eps in pool.imap(run_single_repetition, tasks):
            if config_id not in results_by_config:
                results_by_config[config_id] = []
            results_by_config[config_id].append((result_eval, result_eps))

            if len(results_by_config[config_id]) == n_repetitions:
                results_eval, results_eps = zip(*results_by_config[config_id])
                results_eval = np.array(results_eval)
                results_eps = np.array(results_eps)
                np.savetxt(conf_filename(outdir, param_combinations[config_id], "eval"), results_eval, delimiter=",")
                np.savetxt(conf_filename(outdir, param_combinations[config_id], "eps"), results_eps, delimiter=",")


def create_plot(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval, title, label_params, plotfile,
                plot_eps=False, plot_baseline=False):
    # Create plots for the experiment results
    smoothing_window = 31
    plot = LearningCurvePlot(title)

    for params in param_combinations:
        results_eval = np.loadtxt(conf_filename(outdir, params, "eval"), delimiter=",", ndmin=2)
        if plot_eps:
            results_eps = np.loadtxt(conf_filename(outdir, params, "eps"), delimiter=",", ndmin=2)
            mean_results_eps = np.mean(results_eps, axis=0)
        mean_results_eval = np.mean(results_eval, axis=0)
        conf_results_eval = np.std(results_eval, axis=0) / np.sqrt(n_repetitions)

        plot.add_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), smooth(mean_results_eval,
                       window=smoothing_window), smooth(conf_results_eval, window=smoothing_window),
                       label=", ".join(f"{p}: {params[p]}" for p in label_params))
        if plot_eps:
            plot.add_epsilon_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), mean_results_eps)

    if plot_baseline:
        baseline_results = np.loadtxt("RandomBaselineCartPole.csv", delimiter=",", skiprows=1)
        baseline_results_split = np.split(baseline_results, 2)
        baseline_results_mean = np.mean(baseline_results_split, axis=0)
        baseline_envsteps = baseline_results_mean[:, 2]
        baseline_eval = baseline_results_mean[:, 1]
        plot.add_curve(baseline_envsteps, baseline_eval, label="baseline")

    plot.save(name=plotfile)


if __name__ == '__main__':
    # Define parameter combinations for the experiments
    param_combinations = [
        # DQN with target network and experience replay (closest to SAC params)
        {"gamma": 0.99, "alpha": 0.001, "update_freq": 5, "epsilon": 1,
            "decay_rate": 0.9995, "hidden_dim": 64, "tn": True, "er": True},
    ]

    n_repetitions = 5  # Number of repetitions for each experiment
    n_envsteps = 1000000  # Number of environment steps
    eval_interval = 1000  # Interval for evaluation
    outdir = f"evaluations_{n_envsteps}_envsteps"  # Output directory for results

    run_experiments(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval)
