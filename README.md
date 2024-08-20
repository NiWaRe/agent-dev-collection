# RAG Demo with Weave
This repo aims at demonstraing the continuous development and improvement of a RAG model with Weave. This is a living repo where we'll compare different LLM-based knowledge workers with each other: from a basic LLM, to a basic RAG, over agents with retrieval and function calling, to multi-agent approaaches. 

Check out the Weave Workspace [here](https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/compare-evaluations?evaluationCallIds=%5B%2243c0f1c3-203e-4983-823c-c1134439104a%22%2C%22863675b8-b7a5-4aab-913e-75a34d085f31%22%5D)!

### Getting Started
1. Install `requirements_verbose.txt` in environment (for Mac Silicon)
2. Setup `benchmark.env` in `./config` with necessary API keys (`WANDB_API_KEY`) and optional (`HUGGINGFACEHUB_API_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
3. Set variables accordingly in `general_config.yaml`
    - Set Entity, Project (device for now only CPU)
    - Setup = True the first time to run to extract data and generate dataset
    - The chat model, embedding model, judge model, prompts, params as you want to!
4. Run `main.py`

### Code Structure
- `main.py` - contains the main application flow - serves as an example for bringing everything together
- `setup.py` - contains utility functions for the RAG model `RagModel(weave.Model)` and the data extraction and dataset generation functions
- `evaluatie.py` - contains the `weave.flow.scorer.Scorer` classes to evaluate the correctness, hallucination, and retrieval performance.
- `./configs` - the configs of the project
    - `./configs/benchmark.env` - should contain env vars for your W&B account and the model providers you want to use (HuggingFace, OpenAI, Anthropic, Mistral, etc.)
    - `./configs/requirements.txt` - environment to install necessary dependencies to run RAG
    - `./configs/sources_urls.csv` - a CSV to contain all the Websites and PDFs that should be considered by RAG
    - `./configs/general_config.yaml` - the central config file with models, prompts, params
- `annotate.py` - can be run with `streamlit run annotation.py` to annotate existing datasets or fetch datasets based on production function calls to annotate and save as new dataset.
- `chatbot.py` - can be run with `streamlit run chatbot.py` to serve the RAG Model from Weave and track questions asked to it

