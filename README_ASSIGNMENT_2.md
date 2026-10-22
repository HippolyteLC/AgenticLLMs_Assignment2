# README

This repository includes the modified code to run our custom and self-reflection agents. 

### Experimental prompt / script

1. The config file contains the default parameter settings we used for the local experiments. 
2. The **.sh** files contains the script to run the evaluation code for all desired models. 
Mainly the number of workers, specific local model names, were modified. 
3. To run the local model experiments, inside the github BALROG repository:

Where i is the exercise. 
```bash
chmod +x exercise_i_local_runs.sh
```

```bash 
exercise_i_local_runs.sh
```

### Cloud API Runs (Custom Agent)

The following command was used to evaluate the custom agent on the Gemini cloud API:

To run the evaluation using the Gemini API, ensure you have exported your API key to your environment:
`export GEMINI_API_KEY="api_key"`

```bash
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_text_history=8 \
  envs.names=babyai \
  eval.num_workers=1 \
  eval.num_episodes.babyai=3 \
  eval.max_steps_per_episode=15 \
  client.client_name=gemini \
  client.model_id=gemini-3.5-flash-lite


