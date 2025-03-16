import os
import requests
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
from llm import filter_savr_qnames
from main import compare_distributions
from llamaapi import LlamaAPI
import re

# Initialize the SDK



# Load the Llama model
# llm = Llama(model_path="./models/llama-2-7b.Q4_K_M.gguf", n_ctx=4096)


# prompt = "What is artificial intelligence?"

# # Generate a response
# response = llm(prompt, max_tokens=100)

# print(response["choices"][0]["text"].strip())


LLAMA_API_KEY = "LLAMA_KEY"  # Replace with your actual key
llama = LlamaAPI(LLAMA_API_KEY)

def ask_llama_api(prompt):
    """Send a request to Llama 3-8B API and return the response."""
    api_request_json = {
        "model": "llama3.1-8b",  # Use Llama 3-8B model
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    response = llama.run(api_request_json)

    # Debugging: Print API response
    # print("DEBUG: API Response Status Code:", response.status_code)
    # print("DEBUG: API Response JSON:", response.json())

    try:
        response_data = response.json()
        if isinstance(response_data, list):  # API might return an error list
            print("Unexpected response format: List instead of dict")
            return None
        return response_data["choices"][0]["message"]["content"]
    
    
    
    except KeyError:
        print("Error: Expected keys not found in response.")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None



def ask_llama_to_generate_professional_responses(schema_path, public_path, output_path):
    """Use a local Llama 2 model to generate professional zero-shot responses based on available options and save them to a CSV file."""
    
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    simulated_data = []
    
    for _, row in tqdm(savr_qnames.iterrows(), total=len(savr_qnames), desc="Processing qnames"):
        qname, question = row["qname"], row["question"]
        # print(f"Processing {qname}: {question}")
        
        if qname in public_df.columns:
            data = public_df[qname].dropna()
            response_count = len(data)
            options = sorted(data.unique().tolist())
            if not options or response_count == 0:
                continue
            
            options_dict = {chr(65 + i): option for i, option in enumerate(options)}  
            # print(f"Options: {options_dict}")
            formatted_options = ", ".join([f"{key}: {val}" for key, val in options_dict.items()])
            # print(f"Formatted Options: {formatted_options}")
            
            generated_responses = []
            total_limit = 1000

            with tqdm(total=total_limit, desc=f"Generating responses for {qname}") as pbar:
                while len(generated_responses) < total_limit:
                    batch = 10
                    prompt = (
                        f"Please answer the following survey question with single choice in a strictly structured JSON format.\n\n"
                        f"Example of JSON format: {{\"responses\": [\"A\", \"B\", \"C\"]}}\n\n"
                        f"Question: {question}\nOptions: {formatted_options}\n\n"
                        f"Generate exactly {batch} responses, ensuring that each choice reflects a realistic human decision-making process. "
                        f"Use only the corresponding option letters (A, B, C, etc.) and return the responses in a JSON list format."
                    )
                    # print(f"Prompt: {prompt}")

                    # response = llm(prompt, max_tokens=512)
                    response_text = ask_llama_api(prompt)
                    copy = response_text

                    response_text = re.sub(r'^.*?\{', '{', response_text, flags=re.DOTALL)
                    match = re.search(r'^(.*})', response_text, flags=re.DOTALL)
                    response_text = match.group(1) if match else response_text  # Keep only valid JSON part

                    try:
                        # Extract and parse response from Llama
                        # generated_text = response_text["choices"][0]["text"].strip()
                        # print(generated_text)
                        simulated_responses = json.loads(response_text)
                        # print(f"Generated: {simulated_responses}")
                        # print(f"Generated {len(simulated_responses['responses'])} responses for {qname}")

                        if "responses" in simulated_responses and isinstance(simulated_responses["responses"], list):
                            generated_responses.extend(simulated_responses["responses"])
                            pbar.update(len(simulated_responses["responses"]))
                        else:
                            print(f"Unexpected response format: {simulated_responses}")

                    except Exception as e:
                        print(f"Error parsing Llama response: {e}")
                        print(copy)
                        print(f"response: {response_text}")
                        break
                        return
            
            for resp in generated_responses:
                full_response = options_dict.get(resp, "Unknown")  # Map back to full text
                simulated_data.append({"qname": qname, "question": question, "response": full_response})
    
    simulated_df = pd.DataFrame(simulated_data)
    if not simulated_df.empty:
        simulated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Simulated responses saved to {output_path}")
    else:
        print("Warning: No responses were generated!")


schema_path="./dataset/survey_results_schema.csv"
public_path="./dataset/survey_results_public.csv"
output_path="./results/simulated_responses_llama_0.csv"
ask_llama_to_generate_professional_responses(schema_path, public_path, output_path)# you need to add it
# compare_distributions(public_path, output_path)
