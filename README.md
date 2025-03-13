# May the AI Scientist Locally be with You!

<p align="center">
  <img src="docs/title.jpg">
</p>

<p align="center">
<img src="docs/ai_scientist_with_boy.jpg" width=340px height=192px ><img src="docs/ai_scientist_with_girl.jpg" width=340px height=192px >
</p>

> [!WARNING]
> As for Literature Search APIs, although I set `OpenAlex API` as the default `--engine` argument, there seems to be an error with it (CORE API as well). 
I've already sent an [issue](https://github.com/SakanaAI/AI-Scientist/issues/179) to the authors. But, it may take a while for them or me (I may investigate 
the issue by myself and may be able to provide the solution) to fix the bug. So, if you have your API key for Semantic Scholar API, make sure to use that instead.
And even if you don't have a Semantic Scholar API key, you can still try and see what kind of interactions the codebase provide while looking thourh the standard output.

> [!NOTE]
> This repository accommodates the recently announced reasoning models, such as [**DeepSeek R1**](https://github.com/deepseek-ai/DeepSeek-R1) and [**QwQ**](https://qwenlm.github.io/blog/qwq-32b/), and handles the outputs of those models properly. So, go 
check them out and see it for yourself how they play out in the acamedic-paper-writing field!

## Table of Contents

1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
   - [Docker](#docker)
   - [Anaconda](#anaconda)
3. [Literature Search APIs](#literature-search-apis)
4. [Supported Platforms and Models](#supported-platforms-and-models)
5. [Aider](#aider)
6. [Setting Up the Templates](#setting-up-the-templates)
   - [NanoGPT Template](#nanogpt-template)
   - [2D Diffusion Template](#2d-diffusion-template)
   - [Grokking Template](#grokking-template)
7. [Run AI Scientist Paper Generation Experiments](#run-ai-scientist-paper-generation-experiments)
8. [Getting an LLM-Generated Paper Review](#getting-an-llm-generated-paper-review)
9. [Making Your Own Template](#making-your-own-template)
   - [Community-Contributed Templates](#community-contributed-templates)
10. [Template Resources](#template-resources)
11. [Reference](#reference)

## Introduction
This repository is based on [The AI Scientist](https://github.com/SakanaAI/AI-Scientist), but this repository supports local LLMs models, 
which the original repo does not. It is especially good if you or your organization has enough GPUs for you to use. Some of possible 
advantages are as follows.

- Don't need to send your data to cloud LLMs.
- Once you download a model, you don't need to access the internet to utilize LLMs.
- Can flexibly and locally finetune a LLM model if you want to.
- Don't need to spend money on pay-as-you-go APIs, which are ambiguous and hard to estimate the total costs that you need to pay. Besides, even if you decide to utilize one of the cloud LLMs later on, using local LLMs first can give you an approximation of the costs before using the cloud ones, such as how many tokens a LLM would produce in order to solve your problem. 
- It is also good if you would like to investigate and experiment what kind of outcomes you can expect when you feed your data for internal investigation purposes.

## Environment Setup

This repo has prepared two kinds to create execution environments, Docker and Anaconda. So, pick either one and follow the instruction.
**Note:** Installing `texlive-full` can take a long time.

### Docker

Follow the installation steps below.
```
(Type the following commands at host)
git clone https://github.com/Masao-Taketani/AI-Scientist-with-Local-LLMs.git
cd AI-Scientist-with-Local-LLMs/env_setup
docker build -t ai_scientist .
cd ..
docker run -it --rm --gpus '"device=[device id(s)]"' -v .:/work ai_scientist:latest

(Type the following commands after starting the container)
(Start a screen session in order to start Ollama in another session)
screen -S ollama
ollama serve
(Press [Ctrl+a+d] to get out of the screen session)
ollama pull [ollama model name]
```

### Anaconda

```
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt-get install texlive-full
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Install PyPI requirements
pip install -r requirements.txt

# In another terminal window, start your ollama server
ollama serve

# Coming back to the original terminal, follow the command
ollama pull [ollama model to use for coder]
```

## Literature Search APIs
Since [Semantic Scholar API](https://www.semanticscholar.org/product/api) [does not seem reliable](https://github.com/SakanaAI/AI-Scientist/issues/104), 
the original authors recently added [OpenAlex API](https://docs.openalex.org/how-to-use-the-api/api-overview). This repository additionally add 
[CORE API](https://core.ac.uk/services/api) referring to the [issue](https://github.com/SakanaAI/AI-Scientist/issues/104#issuecomment-2334149214).
Thus, as for a literature search engine, you can pick among `openalex`, `core`, or `semanticscholar` for `--engine` argument. In order to use each one 
of them, you need set an environment variable accordingly.

For OpenAlex API
```
export OPENALEX_MAIL_ADDRESS="Your email address"
```
For CORE API
```
export CORE_API_KEY="Your CORE API key"
```
Semantic Scholar API
```
export S2_API_KEY="Your Semantic Scholar key"
```

## Supported Platforms and Models
For this repo, any models from Ollama and Hugging Face, including the recently announced reasoning models such as [**DeepSeek R1**](https://github.com/deepseek-ai/DeepSeek-R1) and [**QwQ**](https://qwenlm.github.io/blog/qwq-32b/), are supported to be used as local LLMs. 
So, pick a platform between `ollama` and `huggingface` using `--platform` argument. E.g. `--platform huggingface`.

If you'd like to check thought processes when you use one of the reasoning models, use a flag named `--show-thought`. That way, you can see the 
thought processes in the standard outout!

## Aider
One of Ollama models is used for Aider. Thus, pick one model from Ollama using `--coder-ollama-model` argument. 
E.g. `--coder-ollama-model qwen2.5-coder:32b-instruct-q8_0`. Available Ollama models can be referred from the [website](https://ollama.com/). 

## Setting Up the Templates

This section provides instructions for setting up each of the three templates used in the original paper. Before running The AI Scientist experiments, 
please ensure you have completed the setup steps for the templates you are interested in.

### NanoGPT Template

**Description:** This template investigates transformer-based autoregressive next-token prediction tasks.

**Setup Steps:**

1. **Prepare the data:**

   ```bash
   python data/enwik8/prepare.py
   python data/shakespeare_char/prepare.py
   python data/text8/prepare.py
   ```

2. **Create baseline runs (machine dependent):**

   ```bash
   # Set up NanoGPT baseline run
   # NOTE: YOU MUST FIRST RUN THE PREPARE SCRIPTS ABOVE!
   cd templates/nanoGPT
   python experiment.py --out_dir run_0
   python plot.py
   ```

### 2D Diffusion Template

**Description:** This template studies improving the performance of diffusion generative models on low-dimensional datasets.

**Setup Steps:**

1. **Install dependencies:**

   ```bash
   # Set up 2D Diffusion
   git clone https://github.com/gregversteeg/NPEET.git
   cd NPEET
   pip install .
   pip install scikit-learn
   ```

2. **Create baseline runs:**

   ```bash
   # Set up 2D Diffusion baseline run
   cd templates/2d_diffusion
   python experiment.py --out_dir run_0
   python plot.py
   ```

### Grokking Template

**Description:** This template investigates questions about generalization and learning speed in deep neural networks.

**Setup Steps:**

1. **Install dependencies:**

   ```bash
   # Set up Grokking
   pip install einops
   ```

2. **Create baseline runs:**

   ```bash
   # Set up Grokking baseline run
   cd templates/grokking
   python experiment.py --out_dir run_0
   python plot.py
   ```

## Run AI Scientist Paper Generation Experiments

**Note:** Please ensure the setup steps above are completed before running these experiments.

```bash
# Run the paper generation (using Ollama platform for the AI scientist).
python launch_scientist.py --coder-ollama-model qwen2.5-coder:32b-instruct-q8_0 --platform ollama --model qwen2.5:72b-instruct-fp16 --experiment nanoGPT_lite --num-ideas 2
# Run the paper generation (using Hugging Face platform for the AI scientist).
python launch_scientist.py --coder-ollama-model qwen2.5-coder:32b-instruct-fp16 --platform huggingface --model meta-llama/Llama-3.3-70B-Instruct --experiment nanoGPT_lite --num-ideas 2
```

Although I have left `--parallel` argument as the original repo does, I do not recommend to use the argument,
especially if you are using super-large local LLMs, which are recommended to conduct complex tasks such as 
doing academic research, since those take huge GPU resources.

## Getting an LLM-Generated Paper Review

```python
from ai_scientist.llm import init_client_and_model_or_pipe
from ai_scientist.perform_review import load_paper, perform_review

platform = "huggingface" # pick between 'ollama' or 'huggingface'.
model = "Qwen/Qwen2.5-72B-Instruct" # pick a model for the AI scientist

client, model_or_pipe = init_client_and_model_or_pipe(platform, model)

# Load paper from PDF file (raw text)
paper_txt = load_paper("report.pdf")

# Get the review dictionary
review = perform_review(
    paper_txt,
    platform,
    client,
    model_or_pipe,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
review["Overall"]    # Overall score (1-10)
review["Decision"]   # 'Accept' or 'Reject'
review["Weaknesses"] # List of weaknesses (strings)
```

To run batch analysis:

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500 --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## Making Your Own Template

If there is an area of study you would like **The AI Scientist** to explore, it is straightforward to create your own templates. In general, follow the structure of the existing templates, which consist of:

- `experiment.py` — This is the main script where the core content is. It takes an argument `--out_dir`, which specifies where it should create the folder and save the relevant information from the run.
- `plot.py` — This script takes the information from the `run` folders and creates plots. The code should be clear and easy to edit.
- `prompt.json` — Put information about your template here.
- `seed_ideas.json` — Place example ideas here. You can also try to generate ideas without any examples and then pick the best one or two to put here.
- `latex/template.tex` — We recommend using our LaTeX folder but be sure to replace the pre-loaded citations with ones that you expect to be more relevant.

The key to making new templates work is matching the base filenames and output JSONs to the existing format; everything else is free to change.
You should also ensure that the `template.tex` file is updated to use the correct citation style / base plots for your template.

### Community-Contributed Templates

The original repo welcomes community contributions in the form of new templates. While these are not maintained by the original repo, they are delighted to highlight community contributors' templates to others. Below, the original repo lists community-contributed templates along with links to their pull requests (PRs):

- Infectious Disease Modeling (`seir`) - [PR #137](https://github.com/SakanaAI/AI-Scientist/pull/137)
- Image Classification with MobileNetV3 (`mobilenetV3`) - [PR #141](https://github.com/SakanaAI/AI-Scientist/pull/141)
- Sketch RNN (`sketch_rnn`) - [PR #143](https://github.com/SakanaAI/AI-Scientist/pull/143)
- Earthquake Prediction (`earthquake-prediction`) - [PR #167](https://github.com/SakanaAI/AI-Scientist/pull/167)
- Tensorial Radiance Fields (`tensorf`) - [PR #175](https://github.com/SakanaAI/AI-Scientist/pull/175)

*This section is reserved for community contributions. Please submit a pull request to add your template to the list! Please describe the template in the PR description, and also show examples of the generated papers.*

## Template Resources

The original repo provides three templates, which heavily use code from other repositories, credited below:

- **NanoGPT Template** uses code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).
- **2D Diffusion Template** uses code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), and [Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/).
- **Grokking Template** uses code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

I would like to thank the developers of the open-source models and packages for their contributions and for making their work available.

## Reference
[SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
