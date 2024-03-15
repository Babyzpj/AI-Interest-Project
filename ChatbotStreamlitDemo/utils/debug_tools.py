from pprint import pprint as pp


def print_log(func):
    def wrapper(*args, **kwargs):
        # pp(f"Calling {func.__name__} with input: {args}, {kwargs}")
        result = func(*args, **kwargs)
        pp(f"Result of {func.__name__}: {result}")
        return result

    return wrapper


def print_prompt(prompt):
    pp(f"Calling llm with input: {prompt}")
    return prompt


@print_log
def add(a, b):
    return a + b


@print_log
def subtract(a, b):
    return a - b


@print_log
def own_llm():
    add_result = add(5, 3)
    sub_result = subtract(add_result, 3)
    return sub_result


if __name__ == "__main__":
    own_llm()
