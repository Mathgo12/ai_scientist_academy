# Agentic AI Scientist with Academy

## Instructions to run:
- In the **data** folder, run **prepare.py** for each dataset
- Set up an experiment folder in the **templates** directory, or choose an existing one. In the experiment folder, make sure the following files exist:
   1. **experiment.py**: the main script for running the baseline experiment. The argument --out_dir specifies where it should create the folder and save the relevant information from the run.
   2. **latex/template.tex**: pre-loaded citations can be included in the formatting
   3. **plot.py**: takes the information from the run folders and creates plots. 
   4. **prompt.json**: includes information about the template, including "system" and "task_description"
   5. **seed_ideas.jso**n: Place example ideas here.
- Once you are ready, run **run_agents.py** with the relevant options, including the LLM model you want to use and the desired experiment. 
- **run_agents.py** can also be edited to specify agent-specific options like max number of ideas to generate and number of experiment runs.