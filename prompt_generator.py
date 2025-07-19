# writes logic to generate and random prompt/word for user
# reads from data/prompts.txt

import random

def get_random_prompt() -> str:
    """
    Returns a random prompt from the prompts.txt file.
    """
    with open('data/prompts.txt', 'r') as file:
        prompts = file.readlines()
    return random.choice(prompts).strip()