import json
import os
import re
#import backoff
import torch
from transformers import set_seed
import openai


MAX_NUM_TOKENS = 4096

AVAILABLE_PLATFORMS = [
    "huggingface",
    "ollama"
]


# Get N responses from a single message, used for ensembling.
#@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_local_llm(
        msg,
        platform,
        client,
        model_or_pipe,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    content, new_msg_history = [], []
    for _ in range(n_responses):
        c, hist = get_response_from_local_llm(
            msg,
            platform,
            client,
            model_or_pipe,
            system_message,
            print_debug=False,
            msg_history=None,
            temperature=temperature,
        )
        content.append(c)
        new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


#@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_local_llm(
        msg,
        platform,
        client,
        model_or_pipe,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    if 'huggingface' in platform:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        prompt = model_or_pipe.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                *new_msg_history,
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )

        set_seed(0)
        response = model_or_pipe(prompt,
                                 do_sample=True,
                                 temperature=temperature,
                                 max_new_tokens=MAX_NUM_TOKENS,
        )

        content = response[0]["generated_text"][len(prompt):]
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif 'ollama' in platform:
        assert client, "To use an Ollama model, set up the client."
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model_or_pipe,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Platform {platform} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(platform, model):
    if 'ollama' in platform:
        print(f"Using Ollama platform with model {model}.")
        return openai.OpenAI(base_url='http://localhost:11434/v1/',
                             api_key='ollama', # required but ignored
                             )


def init_client_and_model_or_pipe(platform, model):
    if platform == "huggingface":
        from transformers import pipeline
        client = None
        torch_dtype = torch.float16 if "awq" in model.lower() else torch.bfloat16
        pipe = pipeline("text-generation", 
                        model=model, 
                        model_kwargs={"torch_dtype": torch_dtype}, 
                        #device="cuda")
                        device_map="auto")
        model_or_pipe = pipe
    elif platform == "ollama":
        from ai_scientist.llm import create_client
        client = create_client(platform, model)
        model_or_pipe = model
    else:
        raise ValueError(f"Platform {platform} not supported.")
    
    return client, model_or_pipe