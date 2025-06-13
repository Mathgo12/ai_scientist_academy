from academy.behavior import Behavior, action, loop
from academy.exchange.thread import ThreadExchange
from academy.launcher import ThreadLauncher
from academy.manager import Manager
from academy.academy.behavior import Behavior
from academy.academy.handle import Handle

import threading
import time
import openai

import os
import os.path as osp
import json
import argparse

from queue import Queue

from ai_scientist.llm import create_client, AVAILABLE_LLMS, get_response_from_llm, extract_json_between_markers
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import  perform_writeup
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement

class ReviewerAgent(Behavior):
    def __init__(self, results_dir, num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1):
        self.results_dir = results_dir
        self.num_reflections = num_reflections
        self.num_fs_examples = num_fs_examples
        self.num_reviews_ensemble = num_reviews_ensemble
        self.temperature = temperature

        self.ideas_queue = Queue()

        self.model = "gpt-4o-2024-05-13"
        self.client = openai.OpenAI()

    @action
    def upload_idea(self, idea: dict):
        self.ideas_queue.put(idea)

    @loop
    def review(self, shutdown: threading.Event) -> None:
        while not shutdown.is_set():
            if not self.ideas_queue.empty():
                idea = self.ideas_queue.get()
                matching_foldernames = [i for i in os.listdir(self.results_dir) if i.endswith(idea['Name'])]
                for folder_name_end in matching_foldernames:
                    folder_name = osp.join(self.results_dir, folder_name_end)
                    if not osp.exists(f"{folder_name}/{idea['Name']}.pdf"):
                        print(f'WRITEUP MUST BE DONE FIRST FOR {folder_name_end}')
                        continue
                    paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")

                    review = perform_review(
                        paper_text,
                        model=self.model,
                        client=self.client,
                        num_reflections=self.num_reflections,
                        num_fs_examples=self.num_fs_examples,
                        num_reviews_ensemble=self.num_reviews_ensemble,
                        temperature=self.temperature,
                    )

                    with open(osp.join(folder_name, "review.txt"), "w") as f:
                        f.write(json.dumps(review, indent=4))

                    print(f'COMPLETED REVIEW FOR: {folder_name_end}')
            else:
                print('IDEAS MUST BE LOADED')

            time.sleep(10)