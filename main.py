import base64
import requests
import os
from PIL import Image
from dotenv import load_dotenv
import logging
import io


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("Groq API KEY is not set in the .env file")


def process_image(image_path, query):
    try:
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")
            logger.info("Encoded image")


            # Validate image
            img = Image.open(io.BytesIO(image_content))
            img.verify()


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            }
        ]
        return make_api_request("meta-llama/llama-4-scout-17b-16e-instruct", messages)


    except Exception as e:
        logger.error(f"Invalid image format: {str(e)}")
        return {"error": f"Invalid image format: {str(e)}"}


def make_api_request(model, messages):
    responses = {}  


    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )


        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from {model} API: {answer}")
            responses[model] = answer
        else:
            logger.error(f"{model} API Error: {response.status_code} - {response.text}")
            responses[model] = f"Error from {model} API : {response.status_code}"


        return responses


    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return {"error": f"Unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    image_path = "test.png"
    query = "What are the elements in this picture?"
    result = process_image(image_path, query)
    print(result)



