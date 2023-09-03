import json
from llama_cpp import Llama

MODEL_PATH = 'model.bin'
TASKS_PATH = 'fact.json'

PROMPT_TMPL = """\
Decide which of the following summary is more consistent with the article sentence.

Note that consistency means all information in the summary is supported by the article.

Article Sentence: {article}
Summary A: {option_a}
Summary B: {option_b}

The more consistent is Summary"""


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
    llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False) # , n_ctx=n_ctx)
    check_ctx_len(llm)
    for i, task in enumerate(iter_tasks(TASKS_PATH)):
        for swap in (False, True):
            prompt = format_prompt(task, swap)
            print(f'{i}. {prompt}')
            output = llm.create_completion(
                prompt,
                max_tokens=20,
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
