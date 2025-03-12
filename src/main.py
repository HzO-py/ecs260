import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from test import chi_square_test, ks_test, shannon_diversity, simpson_index, kl_divergence
from llm import ask_gpt_to_generate_professional_responses
from RAGinOpenAI import ask_gpt_to_generate_professional_responses_RAG

def compare_distributions(public_path, output_path):
    """Compare the distribution of real vs simulated responses as side-by-side bars and apply statistical tests."""
    real_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    simulated_df = pd.read_csv(output_path, encoding="utf-8", low_memory=False)
    results = []
    
    for qname in tqdm(simulated_df["qname"].unique(), desc="Comparing distributions"):
        if qname in real_df.columns:
            real_data = real_df[qname].dropna()
            sim_data = simulated_df[simulated_df["qname"] == qname]["response"].dropna()

            valid_options = set(real_data.unique())  # Get valid response choices
            sim_data = sim_data[sim_data.isin(valid_options)]  # Keep only valid responses


            if real_data.empty or sim_data.empty:
                continue
            
            # **Centralized Distribution Calculation**
            real_distribution = real_data.value_counts(normalize=True).sort_index()
            sim_distribution = sim_data.value_counts(normalize=True).sort_index()

            # Perform statistical tests
            chi2_stat, chi2_p = chi_square_test(real_distribution, sim_distribution)
            ks_stat, ks_p = ks_test(real_distribution, sim_distribution)

            # Calculate Diversity Indices
            shannon_real, shannon_sim = shannon_diversity(real_distribution), shannon_diversity(sim_distribution)
            simpson_real, simpson_sim = simpson_index(real_distribution), simpson_index(sim_distribution)

            # KL Divergence
            kl_div = kl_divergence(real_distribution, sim_distribution)

            
            print(f"\nStatistical Tests for {qname}:")
            print(f"Chi-square Statistic: {chi2_stat:.4f}, P-value: {chi2_p:.4f}")
            print(f"KS Statistic: {ks_stat:.4f}, P-value: {ks_p:.4f}")

            print(f"\nDiversity Indices:")
            print(f"Shannon Diversity Index - Human: {shannon_real:.4f}, gpt-o4-mini: {shannon_sim:.4f}")
            print(f"Simpson Index - Human: {simpson_real:.4f}, gpt-o4-mini: {simpson_sim:.4f}")

            print(f"\nKL Divergence: {kl_div:.4f}")
            
            labels = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
            real_values = [real_distribution.get(label, 0) for label in labels]
            sim_values = [sim_distribution.get(label, 0) for label in labels]
            
            x = np.arange(len(labels))
            width = 0.4  # Width for bars
            
            ## basic distribution showing
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, real_values, width, label="Human", color="blue", alpha=0.6)
            plt.bar(x + width/2, sim_values, width, label="gpt-o4-mini", color="red", alpha=0.6)
            
            plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
            plt.xlabel("Response Options")
            plt.ylabel("Proportion")
            plt.title(f"Response Distribution for {qname}")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()            # Store results

            results.append({
                "qname": qname,
                "chi2_stat": chi2_stat, "chi2_p": chi2_p,
                "ks_stat": ks_stat, "ks_p": ks_p,
                "shannon_real": shannon_real, "shannon_sim": shannon_sim,
                "simpson_real": simpson_real, "simpson_sim": simpson_sim,
                "kl_divergence": kl_div
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot statistical test results
    plot_statistical_tests(results_df)


def plot_statistical_tests(results_df):
    """Visualize Chi-square, KS test, Shannon diversity, and KL divergence as bar charts."""

    # Extract qname and statistical values
    qnames = results_df["qname"]
    
    # Chi-square and p-values
    chi2_stats = results_df["chi2_stat"]
    chi2_p_vals = results_df["chi2_p"]
    
    # KS test
    ks_stats = results_df["ks_stat"]
    ks_p_vals = results_df["ks_p"]
    
    # Shannon Diversity
    shannon_real = results_df["shannon_real"]
    shannon_sim = results_df["shannon_sim"]

    simpson_real = results_df["simpson_real"]
    simpson_sim = results_df["simpson_sim"]
    
    # KL Divergence
    kl_divergences = results_df["kl_divergence"]

    x = np.arange(len(qnames))  # X-axis positions
    width = 0.35  # Bar width

    # Plot Chi-square statistics and p-values
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chi2_stats, width, label="Chi-square Stat", color="blue", alpha=0.7)
    plt.bar(x + width/2, chi2_p_vals, width, label="Chi-square P-Value", color="red", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Chi-square tests of independence, Statistics and P-Values")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot KS test statistics and p-values
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, ks_stats, width, label="KS Statistic", color="green", alpha=0.7)
    plt.bar(x + width/2, ks_p_vals, width, label="KS P-Value", color="orange", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("KS Test Statistics and P-Values")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot Shannon Diversity Index (Real vs Simulated)
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, shannon_real, width, label="Shannon Index (Human)", color="purple", alpha=0.7)
    plt.bar(x + width/2, shannon_sim, width, label="Shannon Index (gpt-o4-mini)", color="pink", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("Shannon Diversity Index")
    plt.title("Shannon Diversity Index Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot Simpson Diversity Index (Real vs Simulated)
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, simpson_real, width, label="Simpson Index (Human)", color="blue", alpha=0.7)
    plt.bar(x + width/2, simpson_sim, width, label="Simpson Index (GPT-4o-mini)", color="orange", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("Simpson Diversity Index")
    plt.title("Simpson Diversity Index Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot KL Divergence
    plt.figure(figsize=(12, 6))
    plt.bar(x, kl_divergences, width, label="KL Divergence", color="brown", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence Across Questions")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Example usage
schema_path = "./dataset/survey_results_schema.csv"  # Replace with your schema file path
public_path = "./dataset/survey_results_public.csv"  # Replace with your public data file path
output_path="./result/simulated_responses.csv"   # Output file path
key="123"

# ask_gpt_to_generate_professional_responses(schema_path, public_path, output_path, key)# you need to add it
# compare_distributions(public_path, output_path)
ask_gpt_to_generate_professional_responses_RAG(schema_path, public_path, output_path, key)# you need to add it
compare_distributions(public_path, output_path)