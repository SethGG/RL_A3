import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from PolicyNetworkAgent import REINFORCEAgent, ActorCriticAgent
import os
import torch.multiprocessing as mp


def evaluation(agent: REINFORCEAgent):
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
    algo = params["algo"]

    # Create a new environment and agent for each repetition
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    eval_returns = np.zeros(int(n_envsteps / eval_interval))

    if params["algo"] == "REINFORCE":
        agent = REINFORCEAgent(n_actions, n_states, params["alpha"],
                               params["gamma"], params["hidden_dim"], params["normalize"])
    elif algo == "AC":
        agent = ActorCriticAgent(n_actions, n_states, params["alpha"], params["gamma"], params["hidden_dim"],
                                 params["estim_depth"], params["update_episodes"], params["use_advantage"])

    envstep = 0
    eval_num = 0
    while envstep < n_envsteps:
        # Reset the environment
        s, info = env.reset()
        done = False
        trunc = False

        trace_states = []
        trace_actions = []
        trace_rewards = []
        while not done and not trunc:
            trace_states.append(s)
            a = agent.select_action_sample(s)
            trace_actions.append(a)
            s, r, done, trunc, info = env.step(a)
            trace_rewards.append(r)

            envstep += 1
            if envstep % eval_interval == 0:
                # Evaluate the agent periodically
                eval_return = evaluation(agent)
                eval_returns[eval_num] = eval_return
                eval_num += 1

                print(f"Running config: {config_id+1:2}, Repetition {rep_id+1:2}, Environment steps: {envstep:6}, "
                      f"Eval return: {eval_return:3}")

            if envstep == n_envsteps:
                break

        trace_states = np.array(trace_states)
        agent.update(trace_states, trace_actions, trace_rewards)

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
    smoothing_window = 31
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
        # REINFORCE
        {"algo": "REINFORCE", "alpha": 0.001, "gamma": 0.99, "hidden_dim": 64, "normalize": True},
        # A2C
        {"algo": "AC", "alpha": 0.001, "gamma": 0.99, "hidden_dim": 64,
            "estim_depth": 5, "update_episodes": 1, "use_advantage": True},
    ]

    n_repetitions = 5  # Number of repetitions for each experiment
    n_envsteps = 1000000  # Number of environment steps
    eval_interval = 1000  # Interval for evaluation
    outdir = f"evaluations_{n_envsteps}_envsteps"  # Output directory for results

    run_experiments(outdir, param_combinations, n_repetitions, n_envsteps, eval_interval)
