import gradio as gr

from pipeline.extract import zero_shot_generate


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_textbox = gr.Textbox(
                lines=10,
                label="Input Text",
            )
            ontology_dropdown = gr.Dropdown(
                choices=[
                    "https://schema.org",
                ],
                label="Ontology",
            )
            llm_dropdown = gr.Dropdown(
                choices=[
                    ("Llama 3.1 8B", "llama3.1:latest"),
                    ("Llama 3.2 3B", "llama3.2:latest"),
                ],
                label="LLM",
            )
            output_format_dropdown = gr.Dropdown(
                choices=[
                    ("Turtle", "turtle"),
                    ("RDF/XML", "xml"),
                    ("JSON-LD", "json-ld"),
                    ("N-Triples", "ntriples"),
                    ("Notation-3", "n3"),
                ],
                label="Output Graph Format",
            )
            btn = gr.Button(
                value="Generate",
            )
        output_textbox = gr.Code(
            lines=40,
            label="Knowledge Graph",
        )

        btn.click(
            fn=zero_shot_generate,
            inputs=[input_textbox, llm_dropdown, ontology_dropdown, output_format_dropdown],
            outputs=[output_textbox],
        )


if __name__ == "__main__":
    demo.launch()
