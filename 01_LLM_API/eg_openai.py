from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    # model="gpt-5",
    model="gpt-4.1",
    input="오늘 기분이 어때?"
)

print(response.output_text)
