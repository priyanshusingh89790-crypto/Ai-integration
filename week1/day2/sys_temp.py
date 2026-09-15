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
prompt ="suggest me the one word name for the company only 1 name"

message_system ={
    "role": "system",
    "content": "you are content writer and you will suggest me the one word name for the starup of clothing."
}
message = {
    "role": role,
    "content": prompt
}
messages = [message_system, message]
response = client.chat.completions.create(
    model=model, 
    messages=messages,
    temperature =1
)
print(response.choices[0].message.content)
