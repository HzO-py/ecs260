
from llama_cpp import Llama
import json

# Load Llama model (local)

llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf", n_ctx=2048)

def ask_llama_to_generate_responses(question, options, num_responses=1000):
    """Use a local Llama model to generate survey responses."""
    
    formatted_options = ", ".join([f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)])
    
    prompt = (
        f"Please answer the following survey question in a structured JSON format.\n\n"
        f"Question: {question}\nOptions: {formatted_options}\n\n"
        f"Generate exactly {num_responses} responses, ensuring that each choice reflects a realistic human decision-making process. "
        f"Use only the corresponding option letters (A, B, C, etc.) and return the responses in a JSON list format."
    )
    
    output = llm(prompt, max_tokens=2000, stop=["\n"], echo=False)
    
    try:
        response = json.loads(output["choices"][0]["text"])
        if "responses" in response and isinstance(response["responses"], list):
            return response["responses"]
    except Exception as e:
        print(f"Error parsing Llama response: {e}")
    
    return []

# import requests

# # Replace with your actual API key
# API_KEY = "YOUR_LLAMA_API_KEY"

# def ask_llama_api(prompt):
#     """Query Llama API for a response."""
#     url = "https://api.llama-api.com/v1/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "llama-3-8b-chat",  # Use "llama-3-70b-chat" for a larger model
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 150,
#         "temperature": 0.7
#     }

#     response = requests.post(url, headers=headers, json=payload)
    
#     if response.status_code == 200:
#         return response.json()["choices"][0]["message"]["content"]
#     else:
#         print("Error:", response.text)
#         return None

# # Example usage
# response = ask_llama_api("What is AI?")
# print("Llama API Response:", response)






def ask_llama_to_generate_professional_responses(schema_path, public_path, output_path):
    """Use Llama to generate professional zero-shot responses based on available options and save them to a CSV file."""
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    simulated_data = []
    
    for _, row in tqdm(savr_qnames.iterrows(), total=len(savr_qnames), desc="Processing qnames"):
        qname, question = row["qname"], row["question"]
        
        if qname in public_df.columns:
            data = public_df[qname].dropna()
            options = sorted(data.unique().tolist())
            if not options:
                continue
            
            # Ask Llama for simulated responses
            generated_responses = ask_llama_to_generate_responses(question, options, num_responses=2000)
            
            for resp in generated_responses:
                full_response = options_dict.get(resp, "Unknown")
                simulated_data.append({"qname": qname, "question": question, "response": full_response})
    
    simulated_df = pd.DataFrame(simulated_data)
    if not simulated_df.empty:
        simulated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Simulated responses saved to {output_path}")
    else:
        print("Warning: No responses were generated!")
