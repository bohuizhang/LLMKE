import ast
import re

import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff


def response_parser(response):
    """ parse GPT response into a list of strings

    :param response:
    :return:
    """
    response = response.replace("...", "")  # remove ellipsis
    # response = response.splitlines()[0]
    if len(response) > 0:
        if response[0] == " ":
            response = response[1:]
    try:
        response = ast.literal_eval(response)
    except (ValueError, TypeError, SyntaxError):
        if re.search(r'\[\".*\"]', response):  # extract potential list(s) in response
            try:
                response = ast.literal_eval(re.search(r'\[.*]', response).group())
                if type(response) is tuple:  # if multiple lists returned
                    response = response[-1]  # select the last one
            except (ValueError, TypeError, SyntaxError):
                response = [""]
        else:
            response = [""]
    if not response:  # cases return 'None'
        response = [""]
    return response


@retry(
    retry=retry_if_exception_type((openai.error.APIError,
                                   openai.error.APIConnectionError,
                                   openai.error.RateLimitError,
                                   openai.error.ServiceUnavailableError,
                                   openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def get_chat_completion_response(model_name, query, relation, examples=None, context=None):
    if examples:
        messages = [{
            "role": "system",
            "content": "Provide context and examples to the model. The output should follow the exact format as the "
                       "example output, which is a list of strings, such as '['answer_a', 'answer_b]'. Return a list "
                       "with an empty string [\"\"] if no information is available."
        }]
        for q, a in examples.items():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})
    else:
        messages = [
            {"role": "system", "content": "The output should follow the exact format, which is a list of strings, "
                                          "such as \"[\"answer_a\", \"answer_b\"]\". Return a list with an empty "
                                          "string [\"\"] if no information is available."},
            {"role": "user", "content": query}
        ]
    response = openai.ChatCompletion.create(
        temperature=1,
        model=model_name,
        messages=messages
    )
    response = response.choices[0].message.content
    # parse and output if zero-shot or few-shot
    if not context:
        print("Answer is \"" + response + "\"\n")
        response = response_parser(response)
        return response
    # else add context and inference again
    else:
        messages.append({"role": "assistant", "content": response})  # chat history
        # TODO: refine prompt here
        if relation == 'SeriesHasNumberOfEpisodes':
            messages.append({
                "role": "user",
                "content": "Given the context: \"{}\", compared and combined with the previous predictions. "
                           "If IMDb knows, use the information on IMDb. If Wikipedia knows, use the information on "
                           "Wikipedia. Otherwise, use your own knowledge. If there are multiple answers, only return"
                           "one of them. {}".format(context, query)
            })
        else:
            messages.append({
                "role": "user",
                "content": "Given the context: \"{}\", compared and combined with "
                           "the previous predictions, {}".format(context, query)
            })
        # print("Chat history:", messages)
        response = openai.ChatCompletion.create(
            temperature=1,
            model=model_name,
            messages=messages
        )
        response = response.choices[0].message.content
        print("Answer is \"" + response + "\"\n")
        response = response_parser(response)
        return response
