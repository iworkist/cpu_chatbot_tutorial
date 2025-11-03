from openai import OpenAI

client = OpenAI()

models = client.models.list()

# print the model names
for model in models.data:
    print(model.id)
