from llama import ask_llama_to_generate_professional_responses
from RAG_llama import ask_llama_to_generate_rag_responses
from llm import ask_gpt_to_generate_professional_responses
from RAGinOpenAI import ask_gpt_to_generate_professional_responses_RAG
from analysis import compare_distributions,compare_distributions_multiple_models



# Example usage
schema_path = "../dataset/survey_results_schema.csv"  # Replace with your schema file path
public_path = "../dataset/survey_results_public.csv"  # Replace with your public data file path
index_path="../dataset/indexes"  # Replace with your index file path
output_path="../results/RAG_gpt"   # create your Output folder
key="OPENAI_API_KEY"
model_csv_paths = {
    "zeroshot_gpt": "zeroshot_gpt.csv",
    "zeroshot_deepseek": "zeroshot_deepseek.csv",
    "zeroshot_llama": "zeroshot_llama.csv",
    "RAG_gpt": "RAG_gpt.csv",
    "RAG_deepseek": "RAG_deepseek.csv",
    "RAG_llama": "RAG_llama.csv"
}
# Using llama to execute zero-shot prompting task
# ask_llama_to_generate_professional_responses(schema_path, public_path, output_path)

# Using llama to execute RAG  task
# ask_llama_to_generate_rag_responses(schema_path, public_path, output_path, index_path, key)

# Using gpt-4o-mini/deepseek to execute zero-shot prompting task
# ask_gpt_to_generate_professional_responses(schema_path, public_path, output_path, key)

# Using gpt-4o-mini/deepseek to execute RAG task
# ask_gpt_to_generate_professional_responses_RAG(schema_path, public_path, output_path, key)

# compare single distribution
# compare_distributions(public_path, output_path)

# compare multiple distributions
compare_distributions_multiple_models(public_path, model_csv_paths)