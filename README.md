# Any Language GPT-2 Text Generator

This project implements a simple class destined to handle an OpenIA GPT-2 model.

The class generates text from a text-seed. It also implements method for language translation.

It could be used with any language as input text, and the result text could be translated to any language supported by Google Translator.

For more information of the model refer to: ["OpenIA GitHub page"](https://github.com/openai/gpt-2)

## Donwload the model:

Execute:
`python download_model.py`

The model of 124M  of weights will downloaded from the OpenIA repository.

## Class usage:




```
from generative_model import GenerativeModel

# Load a model
gen_model = GenerativeModel(
    model_name='124M',
    seed=0,
    batch_size=1,
    length=None,
    temperature=0.8,
    top_k=0,
    top_p=1,
    models_dir='./models',
    verbose=True)

# Generation step (i.e., from spanish to spanish)
generated_text_v = gen_model.gen_from_sample(
    raw_text='Mi querido hijo, no sabes la alegría que me dió leer tu carta',
    nsamples=1,
    input_lang='es',
    output_lang='es')
    
for text in generated_text_v:
    print(generated_text_v)
```