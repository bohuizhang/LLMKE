from ollama import chat
from ollama import ChatResponse

from pipeline.validate import validate_graph_format


def zero_shot_generate(input_text: str, model_name: str, ontology: str, output_format: str):
    """ Zero shot end-to-end knowledge graph construction from text

    :param input_text:
    :param model_name:
    :param ontology:
    :param output_format:
    :return:
    """
    if model_name in ['llama3.1:latest', 'llama3.2:latest']:
        response: ChatResponse = chat(
            model_name,
            messages=[
                {
                    'role': 'assistant',
                    'content': 'You are a knowledge engineer. You can only output knowledge graphs in the specified format, '
                               'without providing any explanations or additional commentary.',
                },
                {
                    'role': 'user',
                    'content': f'Given the input text: {input_text}, extract the knowledge graph using the ontology {ontology}'
                               f'and convert the resulting graph into the {output_format} format.'
                },
            ],
            stream=False,
            options={
                'temperature': 0.5
            }
        )
        raw_graph_text = response['message']['content']
        validated_graph = validate_graph_format(raw_graph_text, output_format, model_name)
        return validated_graph
    else:
        pass