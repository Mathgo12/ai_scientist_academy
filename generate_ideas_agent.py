from academy.behavior import Behavior, action, loop
from academy.exchange.thread import ThreadExchange
from academy.launcher import ThreadLauncher
from academy.manager import Manager
from academy.academy.behavior import Behavior


import threading
import time

from typing import List, Dict, Union

import requests



import os
import os.path as osp
import json


from queue import PriorityQueue

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers

S2_API_KEY = os.getenv("S2_API_KEY")

class GenerateIdeasAgent(Behavior):
    def __init__(self, base_dir, client, model, max_num_generations=10,
                 num_reflections=3, max_num_considered_ideas=5, max_novelty_iters=10):
        self.base_dir = base_dir
        self.client = client
        self.model = model
        self.max_num_generations = max_num_generations
        self.num_reflections = num_reflections
        self.max_num_considered_ideas = max_num_considered_ideas
        self.engine = 'semanticscholar'

        self.max_novelty_iters = max_novelty_iters

        self.idea_str_archive = []  # list of strings
        self.ideas_pq = PriorityQueue()
        self.num_seed_ideas = 0
        self.seed_ideas_loaded = False

        with open(os.path.join('prompts', 'idea_first_prompt.txt'), "r") as file:
            self.idea_first_prompt = file.read()

        with open(os.path.join('prompts', 'idea_reflection_prompt.txt'), "r") as file:
            self.idea_reflection_prompt = file.read()

        with open(os.path.join('prompts', 'novelty_system_msg.txt'), "r") as file:
            self.novelty_system_msg = file.read()

        with open(os.path.join('prompts', 'novelty_prompt.txt'), "r") as file:
            self.novelty_prompt = file.read()

        with open(osp.join(self.base_dir, "experiment.py"), "r") as f:
            self.code = f.read()

        with open(osp.join(self.base_dir, "prompt.json"), "r") as f:
            self.prompt = json.load(f)

        self.idea_system_prompt = self.prompt["system"]
        self.task_description = self.prompt['task_description']

        self.load_seed_ideas()


    @action
    def load_seed_ideas(self):
        with open(osp.join(self.base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)   # list of dicts
        for seed_idea in seed_ideas:
            self.idea_str_archive.append(json.dumps(seed_idea))
            self.ideas_pq.put((1 / seed_idea['Novelty'], seed_idea))

        print('Loaded Seed Ideas')

        self.seed_ideas_loaded = True
        self.num_seed_ideas = len(self.idea_str_archive)

    @action
    def return_idea(self):

        print('Returned Idea')

        if self.ideas_pq.empty():
            return None
        else:
            idea = self.ideas_pq.get()
            with open(osp.join(self.base_dir, "ideas.json"), "w") as f:
                 json.dump(idea, f, indent=4)

            return idea

    def check_idea_novelty(self, idea: dict):
        print(f"\nChecking novelty of idea: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(self.max_novelty_iters):
            try:
                text, msg_history = get_response_from_llm(
                    self.novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=self.max_novelty_iters,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=self.client,
                    model=self.model,
                    system_message=self.novelty_system_msg.format(
                        num_rounds=self.max_novelty_iters,
                        task_description=self.task_description,
                        code=self.code,
                    ),
                    msg_history=msg_history,
                )

                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                json_output = extract_json_between_markers(text)
                query = json_output["Query"]
                papers = self.search_for_papers(query, result_limit=10, engine=self.engine)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["Novel"] = novel
        return idea


    @loop
    def generate_ideas(self, shutdown: threading.Event) -> None:
        count = 1
        while not shutdown.is_set():
            if self.seed_ideas_loaded and count <= self.max_num_generations:
                print('BEGAN IDEA GENERATION')
                prev_ideas_string = "\n\n".join(self.idea_str_archive)

                msg_history = []
                text, msg_history = get_response_from_llm(
                    self.idea_first_prompt.format(
                        task_description=self.prompt["task_description"],
                        code=self.code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=self.num_reflections,
                    ),
                    client=self.client,
                    model=self.model,
                    system_message=self.idea_system_prompt,
                    msg_history=msg_history,
                )
                print('Response obtained from LLM')
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                # Reflect on ideas
                for j in range(self.num_reflections):
                    text, msg_history = get_response_from_llm(
                        self.idea_reflection_prompt.format(
                            current_round=j + 1, num_reflections=self.num_reflections
                        ),
                        client=self.client,
                        model=self.model,
                        system_message=self.idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (json_output is not None), "Failed to extract JSON from LLM output"

                    if "I am done" in text:
                        break

                print('IDEA GENERATED')
                count += 1

                #idea = json.loads(self.idea_str_archive[-1])
                idea = json_output
                print(idea['Title'])

                print('NOVELTY CHECK')
                idea = self.check_idea_novelty(idea)
                print(f'Novel = {idea["Novel"]}')
                if idea['Novel'] and self.ideas_pq.qsize() < self.max_num_considered_ideas:     # only save the novel ideas
                    self.ideas_pq.put((1/idea['Novelty'], idea))

            elif not self.seed_ideas_loaded:
                print('SEED IDEAS NOT YET LOADED')
            else:
                print('IDEA GENERATION FINISHED')

            time.sleep(10)

    def search_for_papers(self, query, result_limit=10, engine="semanticscholar") -> Union[None, List[Dict]]:
        if not query:
            return None
        if engine == "semanticscholar":
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
                params={
                    "query": query,
                    "limit": result_limit,
                    "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
                },
            )
            print(f"Response Status Code: {rsp.status_code}")
            print(
                f"Response Content: {rsp.text[:500]}"
            )  # Print the first 500 characters of the response content
            rsp.raise_for_status()
            results = rsp.json()
            total = results["total"]
            time.sleep(1.0)
            if not total:
                return None

            papers = results["data"]
            return papers
        else:
            raise NotImplementedError(f"{engine=} not supported!")






