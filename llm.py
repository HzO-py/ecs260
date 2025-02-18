import os
import pandas as pd
import openai
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from scipy.special import rel_entr

def chi_square_test(real_distribution, sim_distribution):
    """Perform Chi-square test of independence between real and simulated responses."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_freqs = np.array([real_distribution.get(cat, 0) for cat in categories])
    sim_freqs = np.array([sim_distribution.get(cat, 0) for cat in categories])

    if np.sum(real_freqs) == 0 or np.sum(sim_freqs) == 0:
        return np.nan, np.nan

    chi2_stat, p_value, _, _ = stats.chi2_contingency([real_freqs, sim_freqs])

    print(f"Chi-square Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}")
    return chi2_stat, p_value


def ks_test(real_distribution, sim_distribution):
    """Perform Kolmogorov-Smirnov test for distribution shape comparison."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_cdf = np.cumsum([real_distribution.get(cat, 0) for cat in categories])
    sim_cdf = np.cumsum([sim_distribution.get(cat, 0) for cat in categories])

    ks_stat = np.max(np.abs(real_cdf - sim_cdf))
    p_value = 1 - ks_stat  # Approximate p-value

    print(f"KS Statistic: {ks_stat:.4f}, P-value: {p_value:.4f}")
    return ks_stat, p_value


def shannon_diversity(distribution):
    """Calculate Shannon's Diversity Index."""
    proportions = np.array(distribution)
    proportions = proportions[proportions > 0]  # Avoid log(0)
    return -np.sum(proportions * np.log2(proportions + 1e-10))


def simpson_index(distribution):
    """Calculate Simpson's Diversity Index."""
    proportions = np.array(distribution)
    return 1 - np.sum(proportions ** 2)


def kl_divergence(real_distribution, sim_distribution):
    """Calculate Kullback-Leibler divergence between real and simulated distributions."""
    
    categories = sorted(set(real_distribution.index).union(set(sim_distribution.index)))
    real_probs = np.array([real_distribution.get(cat, 0) for cat in categories])
    sim_probs = np.array([sim_distribution.get(cat, 0) for cat in categories])

    real_probs /= real_probs.sum()
    sim_probs /= sim_probs.sum()

    real_probs = np.clip(real_probs, 1e-10, 1)
    sim_probs = np.clip(sim_probs, 1e-10, 1)

    return np.sum(rel_entr(real_probs, sim_probs))

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
                full_response = options_dict.get(resp, "Unknown")  # Map back to full text
                simulated_data.append({"qname": qname, "question": question, "response": full_response})
    
    simulated_df = pd.DataFrame(simulated_data)
    if not simulated_df.empty:
        simulated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Simulated responses saved to {output_path}")
    else:
        print("Warning: No responses were generated!")

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
schema_path = "survey_results_schema.csv"  # Replace with your schema file path
public_path = "survey_results_public.csv"  # Replace with your public data file path
output_path = "123"  # Output file path
key='123'

ask_gpt_to_generate_professional_responses(schema_path, public_path, output_path, key)# you need to add it
compare_distributions(public_path, output_path)


