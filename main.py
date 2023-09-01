from llama_cpp import Llama

MODEL_PATH = 'model.bin'

PROMPT_TMPL = """\
Decide which of the following summary is more consistent with the article sentence.

Note that consistency means all information in the summary is supported by the article.

Article Sentence: {article}
Summary A: {option_a}
Summary B: {option_b}

The more consistent is Summary"""


SENT = {
    "article": "and a prized silver dollar each could fetch $ 10 million or more.",
    "correct": "a prized silver dollar each could fetch $ 10 million.",
    "incorrect": "a prized silver dollar each could fetch $ 10 or more.",
}

prompt = PROMPT_TMPL.format(
    article=SENT['article'],
    option_a=SENT['correct'],
    option_b=SENT['incorrect'],
)

n_ctx = 512

llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False, n_ctx=n_ctx)

print(prompt)
n_tokens = len(llm.tokenize(prompt.encode('utf-8')))
print(f'{n_tokens=}')

if n_tokens >= n_ctx:
    print("reloading model to fit context")
    llm = Llama(model_path=MODEL_PATH, n_gqa=8, verbose=False, n_ctx=n_tokens + 20)

output = llm.create_completion(
    # "Q: Name the planets in the solar system? A: ",
    # "Q: How many planets are in solar system? A: ",
    prompt,
    max_tokens=4,
    # stop=["Q:", "\n"],
    stop=["\n"],
    # echo=True,
    # stream=True,
)
print(output)
# for chunk in output:
#     choice = chunk['choices'][0]
#     print(choice['text'], end='')
#     if choice['finish_reason']:
#         print()
#         print(f"finish_reason: {choice['finish_reason']}")
