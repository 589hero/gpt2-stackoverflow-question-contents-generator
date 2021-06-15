import re
import torch
from pydantic import BaseModel, Field
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = 'DHBaek/gpt2-stackoverflow-question-contents-generator'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.to(device)


class TextGenerationInput(BaseModel):
    question_title: str = Field(
        ...,
        title="Question Title",
        description="The question title used to generate question body.",
        max_length=100,
    )
    length: int = Field(
        100,
        title="Max Length",
        description="Max length of the question contents to be generated.",
        ge=50,
        le=150,
    )


class TextGenerationOutput(BaseModel):
    output_1: str
    output_2: str
    output_3: str


def process_text(text):
    text = re.sub('\n{2,}', '\n', text)
    text = text.strip()

    return text


def generate_question_contents(input: TextGenerationInput) -> TextGenerationOutput:
    """Generate question contents based on a given question title."""
    query = f'Title: {input.question_title}, Body: '  # question_title
    input_ids = tokenizer.encode(query, return_tensors='pt')
    input_ids = input_ids.to(device)

    min_length = len(input_ids.tolist()[0])
    max_length = input.length + min_length

    # model generating
    sample_outputs = model.generate(input_ids,
                                    do_sample=True,
                                    min_length=min_length,
                                    max_length=max_length,
                                    no_repeat_ngram_size=2,
                                    num_beams=5,
                                    top_k=40,
                                    top_p=0.95,
                                    num_return_sequences=3)

    question_contents = dict()
    for idx, sample_output in enumerate(sample_outputs):
        question_content = process_text(tokenizer.decode(sample_output, skip_special_tokens=True)[len(query):])

        question_contents[idx] = question_content

    return TextGenerationOutput(output_1=question_contents[0], output_2=question_contents[1], output_3=question_contents[2])
