# The AI Scientist with Local LLMs
This repository is based on [The AI Scientists](https://github.com/SakanaAI/AI-Scientist), but this repository supports local LLMs models, 
which the original repo does not. It is especially good if you or your organization has enough GPUs for you to use. Some of the merits are as follows.

[merits]
- Don't need to send your data to the closed LLMs.
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
Since [Semantic Scholar API](https://www.semanticscholar.org/product/api) does not seem reliable, the original authors recently added 
[OpenAlex API](https://docs.openalex.org/how-to-use-the-api/api-overview). This repository additionally add 
[CORE API](https://core.ac.uk/services/api) referring to the [issue](https://github.com/SakanaAI/AI-Scientist/issues/104#issuecomment-2334149214).
Thus, as for a literature search engine, you can pick among `openalex`, `core`, or `semanticscholar`. 

## Supported Models


## Aider