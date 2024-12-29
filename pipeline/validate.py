from rdflib import Graph

from ollama import chat
from ollama import ChatResponse


def validate_graph_format(graph_text: str, graph_format: str, model_name: str):
    """ Validate the format of the output graph

    :param graph_text:
    :param graph_format:
    :param model_name:
    :return:
    """
    try:
        g = Graph()
        g.parse(data=graph_text, format=graph_format)
        return graph_text
    except Exception as e:
        response: ChatResponse = chat(
            model_name,
            messages=[
                {
                    'role': 'assistant',
                    'content': 'You are a knowledge engineer with expertise in reading and parsing graph data.'
                },
                {
                    'role': 'user',
                    'content': f'The provided graph is not correctly formatted as {graph_format}. Detected error: {e}. '
                               f'Correct the syntax issues in the graph text and return the graph in the proper format,'
                               f' ensuring the graph\'s content remains unchanged. You can only output knowledge graphs'
                               f' in the specified format, without providing any explanations or additional commentary.'
                }
            ],
            stream=False,
            options={
                'temperature': 0.5
            }
        )
        return response['message']['content']