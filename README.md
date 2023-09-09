# Factuality Benchmark

This is my attempt to reproduce results from this article:

https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper

I tried it with a few models, and eventually tuned the prompt to achieve +3% using OpenAssistant 70B model:

- **Accuracy:** `84%`
- **Breakdown:**
    - AB=179 - consistent and correct combination.
    - BA=11 - consistent but incorrect.
    - AA=8 - inconsistent, model biased towards option A.
    - BB=14 - inconsistent, model biased towards option B.

This is just 1% below GPT-4 results.

Model used: [Llama2-70B-OASST with Q5_K_M quantisation](https://huggingface.co/TheBloke/Llama2-70B-OASST-SFT-v10-GGUF)

# Prompt Tuning

## Used template

> <|im_start|>system
> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
> 
> If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
> <|im_end|>
> <|im_start|>user
> Decide which of the following Summary is more consistent with the Article Sentence.
> 
> Note that consistency means all information in the Summary is supported by the Article Sentence.
> 
> Article Sentence: {article}
> Summary Y: {option_a}
> Summary X: {option_b}
> <|im_end|>
> <|im_start|>assistant
> The more consistent is Summary

## Changes summary

1. I used system-user-assistant prompt structure, that was used during model fine-tuning.
2. I changed options labels name from A/B to Y/X to reduce bias towards "A".
3. I prepulated answer with "The more consistent is Summary" to improve conciseness.

# Repo guide

- `fact.py` - script used to run benchmark, saving results to `results.jsonl`
- `anal.ipynb` - Jupyter notebook to analyze the results.
- `results.jsonl` - JSONL with raw model outputs.
