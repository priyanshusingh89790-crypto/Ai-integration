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
prompt1 ="Hi"
prompt2 ="i'm priyanshu"
prompt3 ="and you?"

prompts =[prompt1, prompt2, prompt3]
for prompt in prompts:
    message_system ={
        "role": "system",
        "content": "you are stranger"
    }
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message_system, message]
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        max_tokens=1000,
    )
    usage=response.usage
    print(response.choices[0].message.content)
    print(f"Prompt: {prompt} -->your tokens used: {usage.prompt_tokens}, completion tokens used: {usage.completion_tokens}, total tokens used: {usage.total_tokens} finish reason: {response.choices[0].finish_reason}")

