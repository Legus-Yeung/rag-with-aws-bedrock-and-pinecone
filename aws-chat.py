import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "qwen.qwen3-coder-30b-a3b-v1:0"

user_prompt = "What are Legus's favorite foods and cuisines?"

body = {
    "messages": [
        {"role": "user", "content": user_prompt}
    ],
    "max_tokens": 512
}

response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

response_body = json.loads(response["body"].read())

if "choices" in response_body and len(response_body["choices"]) > 0:
    assistant_content = response_body["choices"][0]["message"]["content"]
    print("Assistant:", assistant_content)
else:
    print("Error: No choices found in response")
    print("Available keys:", list(response_body.keys()))
