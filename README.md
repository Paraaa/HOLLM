


# ⚠️ **_This project is currently a Work in Progress._** ⚠️

HOLLM is under development. Please note that certain features may not be fully implemented or tested yet. Furthermore, setting up the project in the current state may be difficult due to the sparse documentation. All things in this readme can change quickly. The readme might not reflect the current state of the project all the time. For more information take a look into the commits and closed issues.

# HOLLM: Improving LLM-based Global Optimization with Search Space Partitioning

## Installation (Python 3.11)

1. Create a conda environment
   ```
   conda create -n HOLLM python=3.11
   conda activate HOLLM
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set the required API keys as environment variables in your shell configuration (for example, in your ~/.bashrc or ~/.zshrc):

   ```bash
   export OPEN_AI_API_KEY="<YOUR_API_KEY>"
   export GOOGLE_AI_API_KEY=""
   export GROQ_AI_API_KEY=""
   ```

   Reload your terminal or source the configuration file to apply the changes. If you don't have access to the OpenAI API, set all keys to an empty string.

## Huggingface

1. Create an account
2. Create an API key: https://huggingface.co/settings/tokens
3. For some models you need the access rights. You have to accepts the terms of service on the huggingface site for that model and then login via the shell. This has to be done only once. (https://huggingface.co/docs/huggingface_hub/en/guides/cli)
   ```
   huggingface-cli login
   ```
   It will ask you for the API toke you previously created. The key will now be stored in a local cache file and you don't have to do anything anymore.


# How to use HOLLM?
TODO: Explain how to run

To use HOLLM you need to pass a configuration to the `Builder`

```python
from mooLLM.mooLLM import mooLLM
from mooLLM.mooLLM_builder import Builder

config = {...}
mooLLM_builder: Builder = Builder(config=config)
mooLLM: mooLLM = mooLLM_builder.build() # This returns a mooLLM instance
mooLLM.optimize() # Runs the optimization loop
```