import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# api_key = os.getenv("OPENROUTER_API_KEY")
test_model = os.getenv("TEST_MODEL")
base_url = os.getenv("LLM_BASE_URL")
api_key = os.getenv("CEREBRAS_API_KEY")

client = OpenAI(
  # base_url="https://openrouter.ai/api/v1",
  base_url="https://api.cerebras.ai/v1",
  api_key=api_key,
)

response = client.chat.completions.create(
  # model="openai/gpt-5-nano",
  # model="gpt-oss-120b",
  model=test_model,
  messages=[
    {'role': 'user',
     'content': '안녕, 한 문장으로만 답해줘'}
  ],
)

print(response.choices[0].message.content)