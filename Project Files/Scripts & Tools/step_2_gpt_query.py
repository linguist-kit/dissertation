"""
Query a GPT model for responses to prompts, for a specific configuration and set of stimuli.

This file is inspired from part_2_query_gpt.py,
which can be found at: https://github.com/jules-watson/language-ideologies-2024/blob/main/fall_2023_main/part_2_query_gpt.py

Author: Raymond Liu
Date: May 2024
"""

from openai import OpenAI, RateLimitError
from tqdm import trange

import pandas as pd
import csv
import os
import json
import time

from constants import (
    REQUESTS_PER_MINUTE_LIMIT, 
    TOKENS_PER_MINUTE_LIMIT, 
    MODEL_NAME,
    EXPERIMENT_PATH
)
from common import load_json, load_csv


def query_gpt(raw_path, loaded_stimuli, config):
    """
    For each stimuli sentence: queries GPT-3's API for num_generation text responses.
    Unbatched: for potential batching, see
    https://platform.openai.com/docs/guides/batch
    """

    # Initialize variables to track API rate limitations for queries and tokens per minute
    # see https://platform.openai.com/docs/guides/rate-limits for more information
    # see https://platform.openai.com/tokenizer for how gpt-3.5 tokenizer breaks words down into tokens
    queries_avail = REQUESTS_PER_MINUTE_LIMIT
    tokens_avail = TOKENS_PER_MINUTE_LIMIT
    
    query_api_args = config["query_api_args"] # Extract relevant data from configuration
    max_tokens_per_request = query_api_args["max_tokens_per_response"] * query_api_args["num_responses"]

    client = OpenAI()

    start_time = time.time()
    mins = 0 

    with open(raw_path, "w") as f:
        field_names = ["index"] + config["ind_var_cols"] + config["keep_cols"] + ["prompt", "output"]
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writeheader()

        for i in trange(len(loaded_stimuli)):
            prompt = loaded_stimuli.loc[i, "prompt_text"]

            # Repeatedly attempts to query the api until success-
            # (the code should not stop prematurely and result in a loss of progress and tokens,
            # although this try-except does not cover all errors that may occur)
            completed = False

            while not completed:
                try:
                    output = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=query_api_args["max_tokens_per_response"],
                        n=query_api_args["num_responses"],
                    )
                    completed = True
                except RateLimitError as e: 
                    print(f"Index:{i}\n", e, "\n")
                    time.sleep(10)
                    completed = False
                except Exception as e: 
                    print(f"Index:{i}\n", e, "\n")
                    completed = False
            
            # Format raw GPT output into a row in output CSV
            row_dict = {
                "output": output.model_dump_json(), # raw GPT output, formatted as JSON string
                "prompt": prompt,
                "index": i
            } 
            for v in config["ind_var_cols"] + config["keep_cols"]: # add independent variables to output
                row_dict[v] = loaded_stimuli.loc[i, v]
            csv_writer.writerow(row_dict)
            
        
        # Sleep to ensure that request per minute or token per minute limits are not breached
        end_time = time.time()
        queries_avail -= query_api_args["num_responses"]
        tokens_avail -= output.usage.total_tokens
        if queries_avail <= 1 or tokens_avail <= max_tokens_per_request and (end_time - start_time) < 60:
            remaining = 60 - (end_time - start_time)
            remaining = remaining if remaining > 0 else 0
            time.sleep(remaining)
            queries_avail = REQUESTS_PER_MINUTE_LIMIT
            tokens_avail = TOKENS_PER_MINUTE_LIMIT
            mins += 1
            start_time = time.time()
        elif (end_time - start_time) >= 60:
            start_time = time.time()


def process_raw(raw_path, processed_path, config):
    """
    Process the raw data in raw.csv into a cleaner output:
    each response in its own row, the raw model output parsed into separate fields.
    """

    raw_outputs = pd.read_csv(raw_path)
    raw_outputs["output"] = [json.loads(item) for item in raw_outputs["output"]]

    with open(processed_path, 'w') as f:
        fieldnames = ["index"] + config["ind_var_cols"] + config["keep_cols"] + [
            "prompt", "finish_reason", "usage", "response", "id", "object", "created", "model"
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for i in range(len(raw_outputs)):
            output = raw_outputs.loc[i, "output"]
            prompt = raw_outputs.loc[i, "prompt"]

            # Convert a row in the raw output into a row in processed output CSV
            responses = [output["choices"][i]["message"]["content"] for i in range(len(output["choices"]))]
            for response in responses:
                row_dict = {
                    "index": i,
                    "prompt": prompt,
                    "finish_reason": output["choices"][0]["finish_reason"],
                    "usage": dict({
                        "prompt_tokens": output["usage"]["prompt_tokens"], 
                        "completion_tokens": output["usage"]["completion_tokens"],                             
                        "total_tokens": output["usage"]["total_tokens"]
                    }),
                    "response": response,
                    "id": output["id"],
                    "object": output["object"],
                    "created": output["created"],
                    "model": output["model"]
                }
                for v in config["ind_var_cols"] + config["keep_cols"]: # add independent variables to output
                    row_dict[v] = raw_outputs.loc[i, v]
                csv_writer.writerow(row_dict)


def main(config):
    """
    For each stimuli sentence: query the model and save the output of the model.
    """
    print(f"Collecting data from model: {MODEL_NAME}")
    print(f"config: {config}")

    dirname = "/".join(config.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    config_path = f"{dirname}/config.json"
    raw_path = f"{dirname}/{MODEL_NAME}/raw.csv"
    processed_path = f"{dirname}/{MODEL_NAME}/processed.csv"

    if not os.path.exists(f"{dirname}/{MODEL_NAME}"):
        os.mkdir(f"{dirname}/{MODEL_NAME}")

    input_sentences = load_csv(input_path)
    config = load_json(config_path)
    query_gpt(raw_path, input_sentences, config)
    process_raw(raw_path, processed_path, config)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
