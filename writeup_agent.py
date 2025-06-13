from academy.behavior import Behavior, action, loop
from academy.exchange.thread import ThreadExchange
from academy.launcher import ThreadLauncher
from academy.manager import Manager
from academy.academy.behavior import Behavior
from academy.academy.handle import Handle

import threading
import time

from aider.io import InputOutput
from aider.models import Model
from aider.coders import Coder




import os
import os.path as osp
import json
import argparse

from queue import Queue

from ai_scientist.llm import create_client, AVAILABLE_LLMS, get_response_from_llm, extract_json_between_markers
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import  perform_writeup
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement

from review_agent import ReviewerAgent

class WriterAgent(Behavior):
    def __init__(self, results_dir, model, reviewer: Handle[ReviewerAgent], num_cite_rounds=1):
        self.results_dir = results_dir
        self.client, self.client_model = create_client(model)
        self.num_cite_rounds = num_cite_rounds
        self.engine = "semanticscholar"

        self.ideas_queue = Queue()

        if model == "deepseek-coder-v2-0724":
            self.main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            self.main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            self.main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            self.main_model = Model(model)

        self.reviewer = reviewer

    @action
    def upload_idea(self, idea: dict):
        self.ideas_queue.put(idea)

    @loop
    def writeup(self, shutdown: threading.Event) -> None:
        while not shutdown.is_set():
            if not self.ideas_queue.empty():
                idea = self.ideas_queue.get()
                matching_foldernames = [i for i in os.listdir(self.results_dir) if i.endswith(idea['Name'])]
                for folder_name_end in matching_foldernames:
                    folder_name = os.path.join(self.results_dir, folder_name_end)
                    print(folder_name)
                    exp_file = osp.join(folder_name, "experiment.py")
                    notes = osp.join(folder_name, "notes.txt")
                    writeup_file = osp.join(folder_name, "latex", "template.tex")
                    fnames = [exp_file, writeup_file, notes]
                    io = InputOutput(
                        yes=True, chat_history_file=f"{folder_name}/{idea['Name']}_aider.txt"
                    )
                    coder = Coder.create(
                        main_model=self.main_model,
                        fnames=fnames,
                        io=io,
                        stream=False,
                        use_git=False,
                        edit_format="diff"
                    )

                    perform_writeup(idea, folder_name, coder, self.client, self.client_model, engine=self.engine,
                                        num_cite_rounds=self.num_cite_rounds)
                    print('FINISHED IDEA WRITEUP')


                    self.reviewer.action('upload_idea',  idea).result()
                    print('UPLOADED IDEA FOR WRITEUP')

            else:
                print('IDEAS MUST BE LOADED')
            time.sleep(10)