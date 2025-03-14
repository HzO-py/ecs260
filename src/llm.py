import os
import pandas as pd
import openai
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

def filter_savr_qnames(schema_path):
    """Extract qnames where selector is 'SAVR' from schema file (starting from row 18)."""
    schema_df = pd.read_csv(schema_path, encoding="utf-8", low_memory=False)
    
    # Skip first 20 rows
    schema_df = schema_df.iloc[20:]
    
    # Filter rows where 'selector' column is 'SAVR'
    savr_qnames = schema_df[schema_df["selector"] == "SAVR"][["qname", "question"]]
    
    return savr_qnames

def ask_gpt_to_generate_professional_responses(schema_path, public_path, output_path, key):
    """Use GPT-4o to generate professional zero-shot responses based on available options and save them to a CSV file."""
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    client = openai.OpenAI(api_key=key)
    
    simulated_data = []
    
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
            total_limit=2000
            with tqdm(total=total_limit, desc=f"Generating responses for {qname}") as pbar:
                while len(generated_responses) < total_limit:
                    batch = 1000
                    prompt = (
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
                            pbar.update(len(simulated_responses["responses"]))
                        else:
                            print(f"Unexpected response format: {simulated_responses}")
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

