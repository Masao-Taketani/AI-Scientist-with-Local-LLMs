# The AI Scientist with Local LLMs
This repository is based on [The AI Scientists](), but this repository only supports local LLMs models. Thus, there are some merits.
[merits]
- Don't need to send your data to the closed LLMs.
- Don't need to spend money on a pay-as-you-go APIs. It is especially good if your organization has enough GPUs for you to use. It is
also good if you wold like to investigate and experiment what kind of outcomes you can expect when you feed your data before sending it 
to the closed models (internal investigation).

## Installation
```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt install texlive-full

# Install PyPI requirements
pip install -r requirements.txt
```

**Note:** Installing `texlive-full` can take a long time. You may need to [hold Enter](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-some-time-takes-forever) during the installation.

## Supported Models
