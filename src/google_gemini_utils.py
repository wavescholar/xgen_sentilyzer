import time
import requests
import pathlib
import sys
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
from google.cloud import secretmanager
import logging

import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession

project_id = "totemic-tower-407319"
location = "us-central1"
vertexai.init(project=project_id, location=location)


def get_gemini_model(model_name="gemini-1.0-pro"):

    model = GenerativeModel(model_name)
    chat = model.start_chat(response_validation=False)
    return chat


def get_gemini_response(chat: ChatSession, prompt: str, max_retries=3, initial_delay=1):
    """
    Attempts to send a message with retries on failure.
    :chat (ChatSession):
    :param prompt: The message prompt to send.
    :param max_retries: Maximum number of retry attempts.
    :param initial_delay: Initial delay between retries in seconds.
    :return: The response object or None if all retries fail.
    """
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            # Assuming chat.send_message is a function call you're referring to.
            # Replace with the actual function/method call as necessary.
            response = chat.send_message(prompt)
            return_text = response.text

            # We sometimes get garbage added to the result from Gemini for example = <ctrl100>\n滹
            class_names = ["Negative", "Neutral", "Positive"]
            if return_text not in class_names:
                logging.error("The class name {return_text} is not present in the ")
                # Take the garbage out
                if return_text.find("Positive"):
                    return_text = "Positive"
                if return_text.find("Negative"):
                    return_text = "Negative"
                if return_text.find("Neutral"):
                    return_text = "Neutral"

            if return_text is not None:
                return return_text

        except requests.exceptions.RequestException as e:
            logging.error(
                f"An error occurred calling Google Gemini: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff

    logging.error("Google Gemini failed after {max_retries} retries. Giving up.")
    return None


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


# To run you need to be authenticated
def get_api_key(project_id="YOUR_PROJECT_ID", secret_id="YOUR_SECRET_ID"):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the parent name from the project.
    parent = f"projects/{project_id}"

    # Create the parent secret.
    secret = client.create_secret(
        request={
            "parent": parent,
            "secret_id": secret_id,
            "secret": {"replication": {"automatic": {}}},
        }
    )

    # Add the secret version.
    version = client.add_secret_version(
        request={"parent": secret.name, "payload": {"data": b"hello world!"}}
    )

    # Access the secret version.
    response = client.access_secret_version(request={"name": version.name})

    # WARNING: Do not print the secret in a production environment
    payload = response.payload.data.decode("UTF-8")
    print(f"Plaintext: {payload}")


def get_gemini_response_v2(
    api_key="api_key_value", prompt="What is the meaning of life?"
):
    # This needs fixing up - it's not used currently
    try:
        API_KEY = get_api_key(project_id="wavelang", secret_id="")
    except:
        logging.error("could not get GOOGLE_GEMINI_API_KEY")

    genai.configure(api_key=API_KEY)

    logging.info("exiting wavelang driver")

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

    model = genai.GenerativeModel("gemini-pro")

    response = model.generate_content(prompt)

    # print(to_markdown(response.text))
    print(response.text)

    response.prompt_feedback

    response.candidates


if __name__ == "__main__":

    # Test code

    # TODO(developer): Uncomment and set the following variables
    # project_id = 'YOUR_PROJECT_ID'
    # location = 'YOUR_PROJECT_LOCATION'  # Format is 'us-central1'
    # model_id = 'YOUR_MODEL_ID'

    # Instantiate the client.
    # project_id = "YOUR_PROJECT_ID"  # "wavelang"
    # location = "us-central1"
    # vertexai.init(project=project_id, location=location)

    # model = Generative
    project_id = "totemic-tower-407319"  # "wavelang"
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    model = GenerativeModel("gemini-pro")
    chat = model.start_chat()

    def get_chat_response(chat: ChatSession, prompt: str) -> str:
        response = chat.send_message(prompt)
        return response.text

    prompt = "Hello."
    print(get_chat_response(chat, prompt))

    prompt = "What are all the colors in a rainbow?"
    print(get_chat_response(chat, prompt))

    prompt = "Why does it appear when it rains?"
    print(get_chat_response(chat, prompt))
