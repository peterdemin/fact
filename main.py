import sys
from llama_cpp import Llama

# MODEL_PATH = 'oasst.gguf'
MODEL_PATH = 'orca.gguf'


ORCA_TMPL = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Edit the following reading notes to ensure the structure reads well.
Check for grammatical errors and spelling mistakes.
Preserve markdown formatting.

{article}

Answer:
"""

OASST_TMPL = """\
<|im_start|>system
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<|im_end|>
<|im_start|>user
Edit the following reading notes to ensure the structure reads well.
Check for grammatical errors and spelling mistakes.
Preserve markdown formatting.

{article}
<|im_end|>
<|im_start|>assistant
"""

n_ctx = 1000
llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False, n_ctx=n_ctx)
prompt = ORCA_TMPL.format(article=sys.stdin.read())
# print(prompt)
n_tokens = len(llm.tokenize(prompt.encode('utf-8')))
# print(f'{n_tokens=}')
if n_tokens >= n_ctx:
    print("W: Reloading model to fit context: {n_tokens} tokens", file=sys.stderr)
    llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False, n_ctx=n_tokens + 20)

output = llm.create_completion(
    prompt,
    max_tokens=n_tokens * 1.5,
    stop=['<|im_end|>'],
    stream=True,
)
for chunk in output:
    choice = chunk['choices'][0]
    print(choice['text'], end='')
