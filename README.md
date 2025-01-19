# The AI Scientist with Local LLMs

This repository is based on [The AI Scientists](https://github.com/SakanaAI/AI-Scientist), but this repository supports local LLMs models, 
which the original repo does not. It is especially good if you or your organization has enough GPUs for you to use. Some of possible 
advantages are as follows.

- Don't need to send your data to the closed LLMs.
- Once you download a model, you don't need to use the internet to process.
- Can flexibly finetune a LLM model if you want to.
- Don't need to spend money on a pay-as-you-go APIs, which is ambiguous and hard to estimate the total costs that you need to pay.
Besides, even if you decide to utilize one of the closed LLMs later on, using local LLMs first can give you an approximation of the costs 
before using the closed LLMs, such as how many tokens a LLM would produce in order to solve your problem. 
- It is also good if you would like to investigate and experiment what kind of outcomes you can expect when you feed your data for internal 
investigation purpose.


## Environment Setup

This repo has prepared two kinds to create execution environments, Docker and Anaconda. So, pick either one and follow the instruction.
**Note:** Installing `texlive-full` can take a long time.

### Docker

Follow the installation steps below.
```
git clone https://github.com/Masao-Taketani/AI-Scientist-with-Local-LLMs.git
cd AI-Scientist-with-Local-LLMs

(Type the following commands at host)
docker build -t ai_scientist .
docker run -it --rm --ipc=host --gpus '"device=[device id(s)]"' -v .:/work ai_scientist:latest

(Type the following commands after starting the container)
screen -S ollama
ollama serve
(Press [Ctrl+a+d] to get out of the screen session)
ollama pull [ollama model to use for coder]
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
For this repo, any models from Ollama or transformers are supported to be used as local LLMs. So, pick a platform using `--plaform` argument. 
E.g. `--platform transformers`.

## Aider
One of Ollama models is used for Aider. Thus, pick one model from Ollama using `--coder-ollama-model` argument. 
E.g. `--coder-ollama-model qwen2.5-coder:32b-instruct-q8_0`. Available Ollama models can be referred from the [website](https://ollama.com/). 

## Setting Up the Templates

This section provides instructions for setting up each of the three templates used in our paper. Before running The AI Scientist experiments, 
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
# Run the paper generation.
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2
```

Although I have left `--parallel` argument as the original repo does, I do not recommend to use the argument,
especially if you are using super-large local LLMs since those take huge GPU resources.

## Getting an LLM-Generated Paper Review

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from PDF file (raw text)
paper_txt = load_paper("report.pdf")

# Get the review dictionary
review = perform_review(
    paper_txt,
    model,
    client,
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

## Template Resources

We provide three templates, which heavily use code from other repositories, credited below:

- **NanoGPT Template** uses code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).
- **2D Diffusion Template** uses code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), and [Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/).
- **Grokking Template** uses code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

I would like to thank the developers of the open-source models and packages for their contributions and for making their work available.

## Reference
[SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)