"""
Diverse language rules by crossover & mutation
"""
from openai import OpenAI



CROSSOVER_PROMPT="""Please cross over the following prompts and generate a new prompt bracketed with <prompt> and </prompt>.
Prompt 1: {prompt1}
Prompt 2: {prompt2}"""


MUTATE_PROMPT="""Mutate the prompt and generate a new prompt bracketed with <prompt> and </prompt>
Prompt: {prompt}"""



def crossover(
    rule1,
    rule2,
    model,
    base_url,
    api_key
):
    prompt = CROSSOVER_PROMPT.format(prompt1=rule1, prompt2=rule2)
    client = OpenAI(api_key= api_key,
                   base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages = [{"role":"user","content":prompt}],
        temperature=1,
        max_tokens=64
    )
    response = response.choices[0].message.content
    response = response[response.index('<prompt>')+len('<prompt>'):response.index('</prompt>')].strip()
    return response


def mutation(
    rule,
    model,
    base_url,
    api_key
):
    prompt = MUTATE_PROMPT.format(prompt=rule)
    client = OpenAI(api_key= api_key,
                   base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages = [{"role":"user","content":prompt}],
        temperature=1,
        max_tokens=64
    )
    response = response.choices[0].message.content
    response = response[response.index('<prompt>')+len('<prompt>'):response.index('</prompt>')].strip()
    return response