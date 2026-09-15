import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


my_api_key = os.getenv ("Grok_api_key")
if not my_api_key:
    raise ValueError("groq api key not found")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
prompt ="give 5 top llm model name"
message = {
    "role": role,
    "content": prompt
}
messages = [message]
response = client.chat.completions.create(
    model=model, 
    messages=messages
)
print(response.choices[0].message.content)
