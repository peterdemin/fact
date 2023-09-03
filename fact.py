import json
from llama_cpp import Llama

MODEL_PATH = 'model.bin'
TASKS_PATH = 'fact.json'

PROMPT_TMPL = """\
Decide which of the following Summary is more consistent with the Article Sentence.

Note that consistency means all information in the Summary is supported by the Article Sentence.

Article Sentence: {article}
Summary Y: {option_a}
Summary X: {option_b}
Answer: The more consistent is Summary"""


def iter_tasks(filename):
    with open(filename, 'rt', encoding='utf-8') as fobj:
        return json.load(fobj)


def format_prompt(task, swap):
    return PROMPT_TMPL.format(
        article=task['article_sent'],
        option_a=task['incorrect_sent'] if swap else task['correct_sent'],
        option_b=task['correct_sent'] if swap else task['incorrect_sent'],
    )


def check_ctx_len(llm, max_ctx=512):
    for task in iter_tasks(TASKS_PATH):
        prompt = format_prompt(task, False)
        n_tokens = len(llm.tokenize(prompt.encode('utf-8')))
        if n_tokens >= max_ctx:
            raise ValueError(f"Prompt is too long ({n_tokens}): {task}")


def main():
    """
    The more influential parameters/settings on the quality of LLM output are top-p, top-k, temperature, repetition_penalty, and turn templates.
    You can think of Top-p and Top-k that control the “vocabulary size” of the large language models at inference time.
    Since these models predict the next token (word) by calculating the probability of available words, we can control how the model picks the next token when multiple tokens are probable.
    The top-p parameter selects the tokens whose cumulative probability is over a threshold.
    The top-k parameter selects only the k tokens with the top probability.
    With a low top-p value (like 0.15), you allow more rarely used tokens with lower probability to appear, but with a high top-p value (like 0.8) you essentially remove them from the generation vocabulary.
    With a small top-k like 1, you only sample the most probable word; with a larger top-k, you will get more varied results.
    The temperature comes after the probable tokens are selected by top-p or top-k.
    After selecting a pool of potential tokens with top-p or top-k, you can use temperature to control the randomness of the results.
    What temperature does is actually modifies the probability of the tokens — the higher the temperature, the more equal the probability that any of the words in the pool will be drawn is, and thus the more random the result.
    The repetition penalty is a parameter to tell the model how frequently they should use the same token when generating text.
    If the repetition penalty is high, the model is less likely to repeat what it has said in the past or be stuck in a loop repeating the same sentence.
    """
    llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False) # , n_ctx=n_ctx)
    check_ctx_len(llm)
    for i, task in enumerate(iter_tasks(TASKS_PATH)):
        for swap in (False, True):
            prompt = format_prompt(task, swap)
            print(f'{i}. {prompt}')
            output = llm.create_completion(
                prompt,
                max_tokens=20,
                top_k=10,
                top_p=0.9,
                temperature=1e-6,
            )
            answer = output['choices'][0]['text'].strip().split()[0]
            print(answer)
            result = dict(task, answer=answer, swap=swap)
            with open('results.jsonl', 'at', encoding='utf-8') as fobj:
                json.dump(result, fobj)
                fobj.write('\n')


if __name__ == '__main__':
    main()
