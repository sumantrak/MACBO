import os
import argparse
import random
import json
import numpy as np

from utils.logger import Logger
from envs.samplers import Constant
from envs import ErdosRenyi, ScaleFree, BifEnvironment, Dream4Environment
from strategies import MACBOStrategy
from models import DiBS_Linear
from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Experimental Design")
    parser.add_argument(
        "--save_path", type=str, default="results/", help="Path to save result files"
    )
    parser.add_argument(
        "--id", type=str, default=None, help="ID for the run"
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=20,
        help="random seed for generating data (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=20, help="Number of nodes in the causal model"
    )
    parser.add_argument(
        "--reward_node", type=int, default=-1, help="The node we are supposed to optimise"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dag_bootstrap",
        help="Posterior model to use {vcn, dibs, dag_bootstrap}",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Acqusition strategy to use {abcd, random}",
    )
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "--sparsity_factor",
        type=float,
        default=0.0,
        help="Hyperparameter for sparsity regulariser",
    )
    parser.add_argument(
        "--exp_edges",
        type=int,
        default=1,
        help="Number of expected edges in random graphs",
    )
    parser.add_argument(
        "--alpha_lambd", type=int, default=10.0, help="Hyperparameter for the bge score"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data",
    )
    parser.add_argument(
        "--num_starting_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data to start with",
    )
    parser.add_argument(
        "--dibs_steps",
        type=int,
        default=20000,
        help="Total number of steps DiBs to run per training iteration.",
    )
    parser.add_argument(
        "--dibs_graph_prior",
        type=str,
        default='er',
        help="DiBs graph prior (only applicable for bif env).",
    )

    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=3,
        help="Total number of exploration steps in gp-ucb",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Type of noise of causal model",
    )
    parser.add_argument(
        "--bald_temperature", type=float, default=2.0, help="Temperature of soft bald"
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Std of Noise Variables"
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0, help="Mean of Parameter Variables"
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0, help="Std of Parameter Variables"
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0, help="Temperature of Gibbs factor"
    )

    # TODO: improve names
    parser.add_argument('--num_intervention_values', type=int, default=5, help="Number of interventional values to consider.")
    parser.add_argument('--intervention_values', type=float, nargs="+", help='Interventioanl values to set in `grid` value_strategy, else ignored.')
    parser.add_argument('--intervention_value', type=float, default=0.0, help="Interventional value to set in `fixed` value_strategy, else ingored.")

    parser.add_argument(
        "--group_interventions", action='store_true'
    )
    parser.add_argument(
        "--plot_graphs", action='store_true'
    )
    parser.add_argument('--no_sid', action='store_true', default=False)
    parser.set_defaults(group_interventions=False)
    parser.add_argument(
        "--nonlinear", action='store_true', default=False
    )
    parser.add_argument(
        "--value_strategy",
        type=str,
        default="fixed",
        help="Possible strategies: gp-ucb, grid, fixed, sample-dist",
    )
    parser.add_argument(
        "--bif_file", type=str, default="bif/sachs.bif", help="Path of BIF file to load"
    )
    parser.add_argument(
        "--dream4_path", type=str, default="envs/dream4/configurations/", help="Path of DREAM4 files."
    )
    parser.add_argument(
        "--dream4_name", type=str, default="insilico_size10_1", help="Name of DREAM4 experiment to load."
    )
    parser.add_argument(
        "--bif_mapping", type=str, default="{\"LOW\": 0, \"AVG\": 1, \"HIGH\": 2}", help="BIF states mapping"
    )

    parser.set_defaults(nonlinear=False)

    args = parser.parse_args()
    args.node_range = (-10, 10)

    if args.env== "sf":
        args.dibs_graph_prior = "sf"

    if args.reward_node == -1:
        args.reward_node = random.randint(0,args.num_nodes)

    return args


def causal_exps(args):
    print("experiments starting")
    env = ErdosRenyi(
            num_nodes=5,
            exp_edges=2,
            noise_type="isotropic-gaussian",
            noise_sigma=0.1,
            num_samples=10,
            mu_prior=2.0,
            sigma_prior=1.0,
            seed=20,
            nonlinear = False
        )

    model = DiBS_Linear(args)
    env.plot_graph(os.path.join(args.save_path, "graph.png"))
    buffer = ReplayBuffer()
    print(env.sample(args.num_starting_samples))
    buffer.update(env.sample(args.num_starting_samples))
    samples = buffer.data().samples
    args.sample_mean = samples.mean(0)
    args.sample_std = samples.std(0, ddof=1)

    precision_matrix = np.linalg.inv(samples.T @ samples / len(samples))
    model.precision_matrix = precision_matrix
    model.update(buffer.data())
    print(model.dags)
    # print(model.sample(5))
    print(model.sample_interventions([2,3],[2,2],3)[:,:,:,2].mean(2))
    
    print(buffer.data())
    model.update(buffer.data())
    valid_interventions = list(range(args.num_nodes))
    strategy = MACBOStrategy(model,env,args)
    intervention1 = strategy.acquire(valid_interventions, 1)
    
    node1,sampler1 = intervention1
    buffer.update(env.intervene(0, 1, node1, Constant(sampler1), False))
    model.update(buffer.data())
    intervention2 = strategy.acquire(valid_interventions, 1)
    print("interventions")
    print(intervention1)
    print(intervention2)
    node2,sampler2 = intervention2
    avg_reward1 = env.intervene(1,100,node1,Constant(sampler1),False).samples.mean(0)[args.reward_node]
    avg_reward2 = env.intervene(1,100,node2,Constant(sampler2),False).samples.mean(0)[args.reward_node]
    standard_reward = env.sample(1000).samples.mean(0)[args.reward_node]
    print("avg rewards")
    print(avg_reward1)
    print(avg_reward2)
    print("real interventions")
    print(model.sample_interventions([node1],[sampler1],3)[:,:,:,args.reward_node].mean())
    print(model.sample_interventions([node2],[sampler2],3)[:,:,:,args.reward_node].mean())
    print("non intervened")
    print(standard_reward)




if __name__ == "__main__":
    print("Main")
    args = parse_args()
    causal_exps(args)
