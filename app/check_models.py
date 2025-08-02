from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("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"))

models = client.models.list()

for model in models.data:
    print(model.id)
