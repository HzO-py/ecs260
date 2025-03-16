import os
import json
import random
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from llamaapi import LlamaAPI
from llama_index.core import StorageContext, load_index_from_storage
from main import compare_distributions
import openai
import re

# Initialize Llama API with your API key
LLAMA_API_KEY = "LLAMA_API_KEY"  # Replace with your actual key
llama = LlamaAPI(LLAMA_API_KEY)

# Step 1: Load indexes in parallel
def load_single_index(index_folder, i):
    """Loads a single index file if it exists."""
    filename = f"index_{i}.json"
    file_path = os.path.join(index_folder, filename)
    
    if os.path.exists(file_path):
        storage_context = StorageContext.from_defaults(persist_dir=file_path)
        return i, load_index_from_storage(storage_context)
    return None  # Skip if index does not exist

def load_indexes_parallel(index_folder, num_workers=8):
    """Load index files in parallel."""
    index_storage = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_single_index, index_folder, i): i for i in range(1, 158)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Indexes"):
            result = future.result()
            if result:
                i, index_data = result
                index_storage[i] = index_data  # Store in dictionary
    return index_storage

# Step 2: Extract 'SAVR' questions from schema
def filter_savr_qnames(schema_path):
    """Extract qnames where selector is 'SAVR' from schema file."""
    schema_df = pd.read_csv(schema_path, encoding="utf-8", low_memory=False)
    schema_df = schema_df.iloc[20:]  # Skip first 20 rows
    savr_qnames = schema_df[schema_df["selector"] == "SAVR"][["qname", "question"]]
    return savr_qnames

# Step 3: Query Llama 3-8B API
def ask_llama_api(prompt):
    """Send a request to Llama API and return the response."""
    api_request_json = {
        "model": "llama3.1-8b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = llama.run(api_request_json)
    
    try:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except KeyError:
        print("Error: Expected keys not found in response.")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

#  Step 4: Generate Professional Survey Responses using RAG
def ask_llama_to_generate_rag_responses(schema_path, public_path, output_dir, index_folder, key):
    """Use Llama 3-8B with RAG to generate professional responses and save them."""

    openai.api_key = key
    client = openai

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load indexes
    print("Loading indexes... Please wait.")
    index_storage = load_indexes_parallel(index_folder, num_workers=8)
    
    # Extract relevant survey questions
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    simulated_data = []
    
    for _, row in tqdm(savr_qnames.iterrows(), total=len(savr_qnames), desc="Processing qnames"):
        qname, question = row["qname"], row["question"]

        # Check if responses already exist
        csv_path = os.path.join(output_dir, f"{qname}.csv")
        if os.path.exists(csv_path):
            print(f"{qname} already processed. Skipping...")
            continue

        # Extract answer choices from public dataset
        if qname in public_df.columns:
            data = public_df[qname].dropna()
            response_count = len(data)
            options = sorted(data.unique().tolist())
            if not options or response_count == 0:
                continue
            
            options_dict = {chr(65 + i): option for i, option in enumerate(options)}  
            formatted_options = ", ".join([f"{key}: {val}" for key, val in options_dict.items()])
            
            generated_responses = []
            total_limit = 1000  # Number of responses to generate

            with tqdm(total=total_limit, desc=f"Generating responses for {qname}") as pbar:
                while len(generated_responses) < total_limit:
                    batch = 10  # Process in batches to avoid API limits
                    
                    # Select a random index for retrieval
                    random_index = random.choice(list(index_storage.keys()))
                    index = index_storage[random_index]

                    # Query engine for relevant context
                    query_engine = index.as_query_engine(similarity_top_k=3)
                    retrieved_info = query_engine.query(question)

                    # Construct prompt using retrieved knowledge
                    prompt = (
                        f"Using the following retrieved knowledge: {retrieved_info}\n\n"
                        f"Please answer the following survey question in JSON format.\n\n"
                        f"Example of JSON format: {{\"responses\": [\"A\", \"B\", \"C\"]}}\n\n"
                        f"Question: {question}\nOptions: {formatted_options}\n\n"
                        f"Generate exactly {batch} responses, ensuring that each choice reflects a realistic human decision-making process. "
                        f"Use only the corresponding option letters (A, B, C, etc.) and return the responses in a JSON list format."
                    )

                    # Query Llama API
                    response_text = ask_llama_api(prompt)

                    if response_text is None:
                        print(f"Skipping {qname} due to API failure.")
                        break

                    # Ensure valid JSON format
                    response_text = re.sub(r'^.*?\{', '{', response_text, flags=re.DOTALL)
                    match = re.search(r'^(.*})', response_text, flags=re.DOTALL)
                    response_text = match.group(1) if match else response_text  # Keep only valid JSON part
                    # response_text = '{"responses": ' + response_text + '}'

                    try:
                        simulated_responses = json.loads(response_text)
                        if "responses" in simulated_responses and isinstance(simulated_responses["responses"], list):
                            generated_responses.extend(simulated_responses["responses"])
                            pbar.update(len(simulated_responses["responses"]))
                        else:
                            print(f"Unexpected response format: {simulated_responses}")
                    except Exception as e:
                        print(f"Error parsing Llama response: {e}")
                        print(f"Raw response: {response_text}")
                        break
            
            # Save responses
            response_df = pd.DataFrame([{"qname": qname, "question": question, "response": options_dict.get(resp, "Unknown")} for resp in generated_responses])
            response_df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Saved {qname} responses to {csv_path}")

    print("All qnames processed and saved!")

    # Combine all saved responses into a final CSV
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".csv")]
    if not all_files:
        print("No responses were generated. Exiting.")
        return

    final_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    final_output_path = os.path.join(output_dir, "total_results.csv")
    final_df.to_csv(final_output_path, index=False, encoding="utf-8")
    print(f"Final results saved to {final_output_path}")

            # Save results
    #         for resp in generated_responses:
    #             full_response = options_dict.get(resp, "Unknown")  # Map back to full text
    #             simulated_data.append({"qname": qname, "question": question, "response": full_response})
    
    # simulated_df = pd.DataFrame(simulated_data)
    # if not simulated_df.empty:
    #     simulated_df.to_csv(output_path, index=False, encoding="utf-8")
    #     print(f"Simulated responses saved to {output_path}")
    # else:
    #     print("Warning: No responses were generated!")

# Step 5: Run the RAG-based Survey Response Generation
schema_path = "./dataset/survey_results_schema.csv"
public_path = "./dataset/survey_results_public.csv"
output_path = "./results/Simulated_RAG_llama"
index_folder = "./dataset/indexes"
key="OPENAI_API_KEY"
ask_llama_to_generate_rag_responses(schema_path, public_path, output_path, index_folder, key)

# Step 6: Compare with Previous Distributions
# compare_distributions(public_path, output_path)
