from llm import ask_gpt_to_generate_professional_responses
from RAGinOpenAI import ask_gpt_to_generate_professional_responses_RAG
from analysis import compare_distributions,compare_distributions_multiple_models


# Example usage
schema_path = "../dataset/survey_results_schema.csv"  # Replace with your schema file path
public_path = "../dataset/survey_results_public.csv"  # Replace with your public data file path
output_path="../results/RAG_gpt"   # Output folder
key=""
model_csv_paths = {
    "zeroshot_gpt": "zeroshot_gpt.csv",
    "zeroshot_deepseek": "zeroshot_deepseek.csv",
    "zeroshot_llama": "zeroshot_llama.csv",
    "RAG_gpt": "RAG_gpt.csv",
    "RAG_deepseek": "RAG_deepseek.csv",
    "RAG_llama": "RAG_llama.csv"
}
# ask_gpt_to_generate_professional_responses(schema_path, public_path, output_path, key)
# compare_distributions(public_path, output_path)
# ask_gpt_to_generate_professional_responses_RAG(schema_path, public_path, output_path, key)
# compare_distributions(public_path, output_path)
compare_distributions_multiple_models(public_path, model_csv_paths)