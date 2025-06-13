from academy.behavior import Behavior, action, loop
from academy.exchange.thread import ThreadExchange
from academy.launcher import ThreadLauncher
from academy.manager import Manager

from generate_ideas_agent import GenerateIdeasAgent
from experimentation_agent import ExperimenterAgent
from writeup_agent import WriterAgent
from review_agent import ReviewerAgent

import os
import os.path as osp
import json
import argparse

import time

from ai_scientist.llm import create_client, AVAILABLE_LLMS

def main():
    with Manager(
        exchange=ThreadExchange(),  # Replace with other implementations
        launcher=ThreadLauncher(),  # for distributed deployments
    ) as manager:
        parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
        # add type of experiment (nanoGPT, Boston, etc.)
        parser.add_argument(
            "--experiment",
            type=str,
            default="nanoGPT_lite",
            help="Experiment to run AI Scientist on.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gpt-4o-2024-05-13",
            choices=AVAILABLE_LLMS,
            help="Model to use for AI Scientist.",
        )

        parser.add_argument(
            "--skip-idea-generation",
            action="store_true",
            help="Skip idea generation and use existing ideas.",
        )
        parser.add_argument(
            "--check-novelty",
            action="store_true",
            help="Check novelty of ideas.",
        )

        parser.add_argument(
            "--num-cite-rounds",
            type=int,
            default=1,
            help="Number of Citations to Add in Writeup"
        )
        args = parser.parse_args()

        client, client_model = create_client(args.model)

        # OPTIONS TO CHANGE
        MAX_NUM_GENERATIONS = 5     # max number of ideas to generate
        NUM_REFLECTIONS = 2         # number of times to reflect on each idea
        MAX_NUM_CONSIDERED_IDEAS = 3    # max number of ideas in priority queue at any time

        MAX_RUNS = 1       # max number of experiments per idea
        MAX_ITERS = 2      # max number of retries for failed experiments

        base_dir = osp.join("templates", args.experiment)
        results_dir = osp.join("results", args.experiment)

        gen_agent = GenerateIdeasAgent(base_dir=base_dir, client=client, model=client_model, max_num_considered_ideas=MAX_NUM_CONSIDERED_IDEAS, max_num_generations=MAX_NUM_GENERATIONS, num_reflections=NUM_REFLECTIONS)
        gen_agent_handle = manager.launch(gen_agent)

        review_agent = ReviewerAgent(results_dir=results_dir)
        review_agent_handle = manager.launch(review_agent)

        writeup_agent = WriterAgent(results_dir=results_dir, model=args.model, reviewer=review_agent_handle, num_cite_rounds=args.num_cite_rounds)
        writeup_agent_handle = manager.launch(writeup_agent)

        exp_agent = ExperimenterAgent(base_dir=base_dir, results_dir=results_dir, model=args.model, generator=gen_agent_handle, writer=writeup_agent_handle, baseline_run_complete=True, max_runs=MAX_RUNS, max_iters=MAX_ITERS)
        exp_agent_handle = manager.launch(exp_agent)





        time.sleep(200)
        gen_agent_handle.shutdown()
        exp_agent_handle.shutdown()
        writeup_agent_handle.shutdown()
        review_agent_handle.shutdown()




if __name__ == '__main__':
    main()


