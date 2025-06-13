from academy.behavior import Behavior, action, loop
from academy.exchange.thread import ThreadExchange
from academy.launcher import ThreadLauncher
from academy.manager import Manager
from academy.academy.behavior import Behavior
from academy.academy.handle import Handle

import threading
import time
from datetime import datetime
import shutil
import subprocess
from aider.io import InputOutput
from aider.models import Model
from aider.coders import Coder

import os.path as osp
import json

from ai_scientist.perform_experiments import perform_experiments


from generate_ideas_agent import GenerateIdeasAgent
from writeup_agent import WriterAgent

class ExperimenterAgent(Behavior):
    def __init__(self, base_dir,
        results_dir,
        model,     # name (str)
        generator: Handle[GenerateIdeasAgent],
        writer: Handle[WriterAgent],
        max_runs = 5,
        max_iters = 10,
        novelty_threshold = 6,
        baseline_run_complete=False):


        self.base_dir = base_dir
        self.results_dir = results_dir

        if model == "deepseek-coder-v2-0724":
            self.main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            self.main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            self.main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            self.main_model = Model(model)

        self.max_runs = max_runs
        self.max_iters = max_iters
        self.novelty_threshold = novelty_threshold
        self.baseline_run_complete = baseline_run_complete

        self.generator = generator
        self.writer = writer


    @action
    def run_baseline(self):
        command = [
            "python",
            "experiment.py",
            f"--out_dir=run_0",
        ]

        subprocess.run(command, cwd=self.base_dir)
        self.baseline_run_complete = True

    @action
    def fetch_idea(self):
        idea = self.generator.action('return_idea').result()
        if idea['Novelty'] >= self.novelty_threshold:
            return idea
        else:
            return None

    @loop
    def experiments(self, shutdown: threading.Event) -> None:
        while not shutdown.is_set():
            if not self.baseline_run_complete:
                print('RUNNING BASELINE FIRST')
                self.run_baseline()
                print('FINISHED BASELINE RUN')

            idea = self.fetch_idea()   # fetch idea
            if idea is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                idea_name = f"{timestamp}_{idea['Name']}"
                folder_name = osp.join(self.results_dir, idea_name)

                destination_dir = folder_name
                shutil.copytree(self.base_dir, destination_dir, dirs_exist_ok=True)
                with open(osp.join(self.base_dir, "run_0", "final_info.json"), "r") as f:
                    baseline_results = json.load(f)
                # Check if baseline_results is a dictionary before extracting means
                if isinstance(baseline_results, dict):
                    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
                exp_file = osp.join(folder_name, "experiment.py")
                vis_file = osp.join(folder_name, "plot.py")
                notes = osp.join(folder_name, "notes.txt")
                with open(notes, "w") as f:
                    f.write(f"# Title: {idea['Title']}\n")
                    f.write(f"# Experiment description: {idea['Experiment']}\n")
                    f.write(f"## Run 0: Baseline\n")
                    f.write(f"Results: {baseline_results}\n")
                    f.write(f"Description: Baseline results.\n")

                # create Coder object
                fnames = [exp_file, vis_file, notes]
                io = InputOutput(
                    yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
                )

                # (one Coder object for every idea)
                coder = Coder.create(
                    main_model=self.main_model,
                    fnames=fnames,
                    io=io,
                    stream=False,
                    use_git=False,
                    edit_format="diff",
                )
                # Copy perform_experiments code and run_experiment code
                print(f'STARTING EXPERIMENTS FOR IDEA: {idea["Title"]}')
                success = perform_experiments(idea, folder_name, coder, baseline_results, max_runs=self.max_runs, max_iters=self.max_iters)
                if not success:
                    print(f"Experiments failed for idea {idea_name}")

                print(f'EXPERIMENTS COMPLETE FOR IDEA: {idea["Title"]}')

                self.writer.action('upload_idea', idea).result()
                print(f'SENT IDEA FOR WRITEUP: {idea["Title"]}')


            else:
                print('WATING FOR AN IDEA')

            time.sleep(10)