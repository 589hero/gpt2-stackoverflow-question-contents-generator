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
        description="The Question title used to generate question body.",
        max_length=30,
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


def postprocessing(text):
    text = re.sub('\n{2,}', '\n', text)

    return text


def generate_question_contents(input_ids, min_length, max_length=100, num_return_sequences=1):
    try:
        # model generating
        sample_outputs = model.generate(input_ids,
                                        do_sample=True,
                                        min_length=min_length,
                                        max_length=max_length,
                                        no_repeat_ngram_size=2,
                                        num_beams=5,
                                        top_k=40,
                                        top_p=0.95,  # 누적 확률이 95%인 후보집합에서만 생성
                                        num_return_sequences=num_return_sequences)

        result = dict()

        for idx, sample_output in enumerate(sample_outputs):
            question_body = tokenizer.decode(sample_output, skip_special_tokens=True)

            result[idx] = question_body

        return result

    except Exception as e:
        print(f'Error occur : {e}')
        return {'error': e}


def get_question_contents(input: TextGenerationInput) -> TextGenerationOutput:
    """Generate question body based on a given question title."""
    query = f'Title: {input.question_title}, Body: ' # question_title
    input_ids = tokenizer.encode(query)
    min_length = len(input_ids.tolist()[0])
    max_length = input.length + min_length

    results = generate_question_contents(input_ids, min_length, max_length, num_return_sequences=3)

    text = dict()
    for idx, result in enumerate(results):
        text[idx] = postprocessing(tokenizer.decode(results[idx], skip_special_tokens=True)[len(query):])

    return TextGenerationOutput(output_1=text[0], output_2=text[1], output_3=text[2])
