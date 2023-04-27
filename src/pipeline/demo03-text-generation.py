from transformers import pipeline, set_seed
"""
将此模型直接与管道一起使用以生成文本
"""
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
