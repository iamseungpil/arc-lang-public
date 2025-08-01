import random
import string


def random_str(k: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=k))
