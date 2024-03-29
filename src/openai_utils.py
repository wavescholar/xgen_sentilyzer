import os
import logging
import time
from openai import OpenAI
import requests

logger = logging.getLogger("MainLogger")


OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY")
if OPENAI_APIKEY is None:
    raise ValueError("OPENAI_APIKEY environment variable not set")

openai_client = OpenAI(api_key=OPENAI_APIKEY)


def get_openai_response(
    system_prompt,
    sentence,
    openai_model="gpt-3.5-turbo",
    max_retries=3,
    initial_delay=1,
):
    """
    :param max_retries: Maximum number of retry attempts.
    :param initial_delay: Initial delay between retries in seconds.
    :return: The response object or None if all retries fail.
    logger.info("system_prompt length= " + str(len(system_prompt)))
    logger.info("sentence length = " + str(len(sentence)))
    """

    prompt_content_openai = system_prompt + sentence

    prompt_openai = [
        {
            "role": "user",
            "content": prompt_content_openai,
        }
    ]
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            # Assuming chat.send_message is a function call you're referring to.
            # Replace with the actual function/method call as necessary.
            response = openai_client.chat.completions.create(
                model=openai_model,
                messages=prompt_openai,
                max_tokens=1,
                temperature=0.25,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"],
            )
            msg = response.choices[0].message.content
            return msg
        except requests.exceptions.RequestException as e:
            logging.error(
                f"An error occurred calling OpenAI: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff

    logging.error("OpenAI failed after {max_retries} retries. Giving up.")
    return None
