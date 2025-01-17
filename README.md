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


## Installation
```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt install texlive-full

# Install PyPI requirements
pip install -r requirements.txt
```

**Note:** Installing `texlive-full` can take a long time.


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

## Supported Models


## Aider