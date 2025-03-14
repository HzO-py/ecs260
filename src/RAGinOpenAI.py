import os
import pandas as pd
import openai
import matplotlib.pyplot as plt
import json
import numpy as np
import random
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader,GPTVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import concurrent.futures

def load_single_index(index_folder, i):
    """Loads a single index file if it exists."""
    filename = f"index_{i}.json"
    file_path = os.path.join(index_folder, filename)
    
    if os.path.exists(file_path):
        storage_context = StorageContext.from_defaults(persist_dir=file_path)
        return i, load_index_from_storage(storage_context)
    return None  # Return None if file does not exist

def load_indexes_parallel(index_folder, num_workers=8):
    """Loads index files in parallel using ThreadPoolExecutor."""
    index_storage = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_single_index, index_folder, i): i for i in range(1, 158)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Indexes"):
            result = future.result()
            if result:
                i, index_data = result
                index_storage[i] = index_data  # Store in dictionary if successfully loaded

    return index_storage

def filter_savr_qnames(schema_path):
    """Extract qnames where selector is 'SAVR' from schema file (starting from row 18)."""
    schema_df = pd.read_csv(schema_path, encoding="utf-8", low_memory=False)
    
    # Skip first 20 rows
    schema_df = schema_df.iloc[20:]
    
    # Filter rows where 'selector' column is 'SAVR'
    savr_qnames = schema_df[schema_df["selector"] == "SAVR"][["qname", "question"]]
    
    return savr_qnames

def ask_gpt_to_generate_professional_responses_RAG(schema_path, public_path, output_path, key):
    """Use GPT-4o to generate professional zero-shot responses based on available options and save them to a CSV file."""
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    # client = openai.OpenAI(api_key=key)
    openai.api_key = key
    client = openai
    simulated_data = []

    print('Please wait for a long time to load indexes.....')
    index_storage = load_indexes_parallel("../dataset/indexes", num_workers=8)  # It will take a long time!!!!!
    
    for _, row in tqdm(savr_qnames.iterrows(), total=len(savr_qnames), desc="Processing qnames"):
        qname, question = row["qname"], row["question"]

        os.makedirs(output_path, exist_ok=True)
        pkl_path = os.path.join(output_path, f"{qname}.pkl")
        if os.path.exists(pkl_path):
            print(f"{qname} already processed. Skipping...")
            continue
        
        if qname in public_df.columns:
            data = public_df[qname].dropna()
            response_count = len(data)
            options = sorted(data.unique().tolist())
            if not options or response_count == 0:
                continue
            
            options_dict = {chr(65 + i): option for i, option in enumerate(options)}  
            formatted_options = ", ".join([f"{key}: {val}" for key, val in options_dict.items()])
            
            generated_responses = []
            total_limit=1
            with tqdm(total=total_limit, desc=f"Generating responses for {qname}") as pbar:
                while len(generated_responses) < total_limit:
                    batch = 1

                    while True:
                        random_value = random.randint(1, 157)
                        if random_value in index_storage:  
                            index = index_storage[random_value]
                            break  

                    # Initialize the query engine
                    query_engine = index.as_query_engine(similarity_top_k=3)
                    retrieved_info = query_engine.query(question)

                    prompt = (
                        f"Using the following information: {retrieved_info}.\n\n"
                        # it's relevant to the filed {mapped_name}
                        f"Please answer the following survey question in a structured JSON format.\n\n"
                        f"Question: {question}\nOptions: {formatted_options}\n\n"
                        f"Generate exactly {batch} responses, ensuring that each choice reflects a realistic human decision-making process. "
                        f"Use only the corresponding option letters (A, B, C, etc.) and return the responses in a JSON list format."
                    )

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    
                    try:
                        simulated_responses = json.loads(response.choices[0].message.content)
                        if "responses" in simulated_responses and isinstance(simulated_responses["responses"], list):
                            generated_responses.extend(simulated_responses["responses"])
                        else:
                            print(f"Unexpected response format: {simulated_responses}")
                        pbar.update(len(simulated_responses["responses"]))

                    except Exception as e:
                        print(f"Error parsing GPT response: {e}")
            
            for resp in generated_responses:
                if isinstance(resp, dict): 
                    resp = resp.get("option", "Unknown")
                full_response = options_dict.get(resp, "Unknown")  # Map back to full text
                simulated_data.append({"qname": qname, "question": question, "response": full_response})
                simulated_df = pd.DataFrame(simulated_data)

            if not simulated_df.empty:
                try:
                    simulated_df.to_pickle(pkl_path)
                    print(f"Saved {qname} data to {pkl_path}")
                except Exception as e:
                    print(f"Failed to save {qname} data: {e}")

    print("All qnames processed and saved!")

    all_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".pkl")]
    df_list = [pd.read_pickle(f) for f in all_files]
    final_df = pd.concat(df_list, ignore_index=True)
    output_path=os.path.join(output_path,'total_results.csv')
    final_df.to_csv(output_path, index=False, encoding="utf-8")

    return
