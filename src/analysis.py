import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
from test import chi_square_test, ks_test, shannon_diversity, simpson_index, kl_divergence

from llm import filter_savr_qnames

def distribution_show(real_distribution,sim_distribution,qname):
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

def compare_distributions(public_path, output_path):
    """Compare the distribution of real vs simulated responses as side-by-side bars and apply statistical tests."""
    real_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    simulated_df = pd.read_csv(os.path.join(output_path,'total_results.csv'), encoding="utf-8", low_memory=False)
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

            # distribution_show(real_distribution,sim_distribution,qname)

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
    plt.bar(x + width/2, shannon_sim, width, label="Shannon Index (LLM)", color="pink", alpha=0.7)
    plt.xticks(ticks=x, labels=qnames, rotation=45, ha="right")
    plt.ylabel("Shannon Diversity Index")
    plt.title("Shannon Diversity Index Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot Simpson Diversity Index (Real vs Simulated)
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, simpson_real, width, label="Simpson Index (Human)", color="blue", alpha=0.7)
    plt.bar(x + width/2, simpson_sim, width, label="Simpson Index (LLM)", color="orange", alpha=0.7)
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

def plot_distribution_for_qname(q_label, qname, real_distribution, model_distributions, model_order=None, save_dir="../results"):
    """
    参数：
      - qname: 问题名称，用于图表标题与文件名
      - real_distribution: pd.Series，人类回答分布（索引为选项，值为频率）
      - model_distributions: dict，键为模型名称，值为该模型的回答分布（pd.Series）
      - model_order: 指定显示顺序，若为 None 则按字典顺序显示
      - save_dir: 保存图片的目录
    """
    if model_order is None:
        model_order = list(model_distributions.keys())
    
    # 统一所有出现过的选项（并集），对缺失选项补 0
    all_options = set(real_distribution.index)
    for dist in model_distributions.values():
        all_options = all_options.union(dist.index)
    all_options = sorted(list(all_options))

    enumerated_options = [(f"Choice {i}", opt) for i, opt in enumerate(all_options, start=1)]
    short_labels = [pair[0] for pair in enumerated_options]   # ["Choice 1", "Choice 2", ...]
    original_labels = [pair[1] for pair in enumerated_options]# ["Bad at...", "Good, but...", ...]

    
    human_dist = real_distribution.reindex(all_options, fill_value=0)
    model_dist_filled = {m: model_distributions[m].reindex(all_options, fill_value=0)
                         for m in model_order}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_options))
    total_groups = len(model_order) + 1  # 人类 + 各模型
    bar_width = 0.8 / total_groups
    offset = - (total_groups - 1) / 2 * bar_width  # 使所有柱子居中显示

    # 绘制人类分布（蓝色）
    ax.bar(x + offset, human_dist.values, bar_width, label="Human", color="blue", alpha=0.7)
    
    # 绘制各模型分布
    for i, m in enumerate(model_order):
        current_offset = offset + (i + 1) * bar_width
        ax.bar(x + current_offset, model_dist_filled[m].values, bar_width, label=m, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{q_label}: {qname}")
    ax.legend()
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{q_label}_{qname}_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined distribution for {qname} at {save_path}")

    return enumerated_options


def plot_all_qname_distributions(qname_dists, save_dir="../results"):
    """
    遍历 qname_dists 中保存的每个 qname 的分布数据，绘制合并图。
    qname_dists 的结构为：
      { qname: { "human": pd.Series, "models": { model_name: pd.Series, ... } }, ... }
    """
    qnames_options=[]
    for idx, (qname, dist_data) in enumerate(qname_dists.items(), start=1):
        model_distributions = dist_data.get("models", {})
        if not model_distributions:
            continue
        # 可选：指定模型显示顺序
        model_order = ["zeroshot_gpt", "zeroshot_deepseek", "zeroshot_llama",
                       "RAG_gpt", "RAG_deepseek", "RAG_llama"]
        # 保留已有的模型
        model_order = [m for m in model_order if m in model_distributions]
        options=plot_distribution_for_qname(f"Q{idx}", qname, dist_data["human"], model_distributions, model_order, save_dir=save_dir)
        qnames_options.append({
            "q_label": f"Q{idx}",
            "qname": qname,
            "options": options
        })

    schema_df = filter_savr_qnames("../dataset/survey_results_schema.csv")
    question_map = {row.qname: row.question for _, row in schema_df.iterrows()}

    question_rows = []
    choice_rows = []

    for item in qnames_options:
        q_label = item["q_label"]
        qname = item["qname"]
        question = question_map.get(qname, "")
        question_rows.append([q_label, qname, question])
        for choice_label, orig_label in item["options"]:
            choice_rows.append([q_label, choice_label, orig_label])

    question_csv = os.path.join(save_dir, "question_map.csv")
    with open(question_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q_label", "qname", "question"])
        writer.writerows(question_rows)
    print(f"Saved question mapping CSV at: {question_csv}")

    # 保存选项映射 CSV
    choice_csv = os.path.join(save_dir, "choice_map.csv")
    with open(choice_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q_label", "ChoiceLabel", "OriginalOption"])
        writer.writerows(choice_rows)
    print(f"Saved choice mapping CSV at: {choice_csv}")

def compare_distributions_multiple_models(public_path, model_csv_paths):
    """
    参数：
    -------
    public_path : str
        人类答案CSV文件的路径
    model_csv_paths : dict
        键为模型名称(如 'zeroshot_gpt')，值为对应CSV文件路径
    """
    real_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    results = []
    qname_dists = {}
    
    # 遍历不同模型+prompt的CSV文件
    for model_name, csv_path in model_csv_paths.items():
        simulated_df = pd.read_csv(os.path.join('../results',csv_path), encoding="utf-8", low_memory=False)
        
        # 遍历问题 qname
        for qname in tqdm(simulated_df["qname"].unique(), desc=f"Comparing distributions for {model_name}"):
            if qname in real_df.columns:
                real_data = real_df[qname].dropna()
                sim_data = simulated_df[simulated_df["qname"] == qname]["response"].dropna()

                # 1) 在每个qname中随机只读取1000份答案
                if len(sim_data) > 1000:
                    sim_data = sim_data.sample(n=1000, random_state=42)

                # 只保留两者共有的选项，防止出现无效选项
                valid_options = set(real_data.unique())
                sim_data = sim_data[sim_data.isin(valid_options)]

                if real_data.empty or sim_data.empty:
                    continue

                # 计算分布
                real_distribution = real_data.value_counts(normalize=True).sort_index()
                sim_distribution = sim_data.value_counts(normalize=True).sort_index()

                if qname not in qname_dists:
                    qname_dists[qname] = {"human": real_distribution, "models": {}}
                qname_dists[qname]["models"][model_name] = sim_distribution

                # 统计检验
                chi2_stat, chi2_p = chi_square_test(real_distribution, sim_distribution)
                ks_stat, ks_p = ks_test(real_distribution, sim_distribution)

                # 多样性指数
                shannon_real = shannon_diversity(real_distribution)
                shannon_sim = shannon_diversity(sim_distribution)
                simpson_real = simpson_index(real_distribution)
                simpson_sim = simpson_index(sim_distribution)

                # KL 散度
                kl_div = kl_divergence(real_distribution, sim_distribution)

                results.append({
                    "model_name": model_name,
                    "qname": qname,
                    "chi2_stat": chi2_stat, 
                    "chi2_p": chi2_p,
                    "ks_stat": ks_stat, 
                    "ks_p": ks_p,
                    "shannon_real": shannon_real, 
                    "shannon_sim": shannon_sim,
                    "simpson_real": simpson_real, 
                    "simpson_sim": simpson_sim,
                    "kl_divergence": kl_div
                })

    results_df = pd.DataFrame(results)
    # 绘图
    plot_statistical_tests_multiple_models(results_df)
    plot_all_qname_distributions(qname_dists, save_dir="../results/distribution")


def plot_statistical_tests_multiple_models(results_df):
    """
    生成 5 张图：
      1) Chi-square(stat+p) -> chi-square.png
      2) KS(stat+p) -> ks.png
      3) Shannon(对比人类 vs LLM) -> shannon.png
      4) Simpson(对比人类 vs LLM) -> simpson.png
      5) KL Divergence -> kl.png
    
    其中每张图是2行x3列的子图，行首标注zeroshot/RAG，列顶标注gpt/deepseek/llama
    去掉每个子图内的模型名标题。
    """

    # 2) 行首和列顶标签
    row_labels = ["zeroshot", "RAG"]
    col_labels = ["GPT4o-mini", "Deepseek", "llama"]

    # 需要保证 models_in_order 的顺序与行列标签对应
    models_in_order = [
        "zeroshot_gpt", "zeroshot_deepseek", "zeroshot_llama",
        "RAG_gpt", "RAG_deepseek", "RAG_llama"
    ]

    # 为了绘制更简洁，定义一个辅助函数：
    def get_row_col_axes(nrows=2, ncols=3, figsize=(18,10)):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
        return fig, axes

    # -------------------
    # 1) Chi-square
    # -------------------
    fig_chi, axes_chi = get_row_col_axes()
    axes_chi = np.array(axes_chi)  # 2D

    for i, model_name in enumerate(models_in_order):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes_chi[row_idx, col_idx]

        # 行首标注
        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=18)
        # 列顶标注
        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=18)

        sub_df = results_df[results_df["model_name"] == model_name].copy()
        x = np.arange(len(sub_df))
        labels = [f"Q{i+1}" for i in range(len(sub_df))]
        ax.bar(x, sub_df["chi2_p"], width=0.5, color="red", alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Question")
        ax.set_ylabel("p-value")       # 改成 p-value
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/chi-square.png")

    # -------------------
    # 2) KS
    # -------------------
    fig_ks, axes_ks = get_row_col_axes()
    axes_ks = np.array(axes_ks)

    for i, model_name in enumerate(models_in_order):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes_ks[row_idx, col_idx]

        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=18)
        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=18)

        sub_df = results_df[results_df["model_name"] == model_name].copy()
        x = np.arange(len(sub_df))
        labels = [f"Q{i+1}" for i in range(len(sub_df))]
        ax.bar(x, sub_df["ks_p"], width=0.5, color="orange", alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Question")
        ax.set_ylabel("p-value")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/ks.png")

    # -------------------
    # 3) Shannon
    # -------------------
    fig_shannon, axes_shannon = get_row_col_axes()
    axes_shannon = np.array(axes_shannon)

    for i, model_name in enumerate(models_in_order):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes_shannon[row_idx, col_idx]

        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=18)
        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=18)

        sub_df = results_df[results_df["model_name"] == model_name].copy()
        x = np.arange(len(sub_df))
        labels = [f"Q{i+1}" for i in range(len(sub_df))]
        ax.bar(x - 0.2, sub_df["shannon_real"], width=0.4, label="Shannon(Human)", color="purple", alpha=0.7)
        ax.bar(x + 0.2, sub_df["shannon_sim"], width=0.4, label="Shannon(LLM)", color="pink", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Question")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/shannon.png")

    # -------------------
    # 4) Simpson
    # -------------------
    fig_simpson, axes_simpson = get_row_col_axes()
    axes_simpson = np.array(axes_simpson)

    for i, model_name in enumerate(models_in_order):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes_simpson[row_idx, col_idx]

        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=18)
        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=18)

        sub_df = results_df[results_df["model_name"] == model_name].copy()
        x = np.arange(len(sub_df))
        labels = [f"Q{i+1}" for i in range(len(sub_df))]
        ax.bar(x - 0.2, sub_df["simpson_real"], width=0.4, label="Simpson(Human)", color="blue", alpha=0.7)
        ax.bar(x + 0.2, sub_df["simpson_sim"], width=0.4, label="Simpson(LLM)", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Question")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/simpson.png")


    # -------------------
    # 5) KL
    # -------------------
    fig_kl, axes_kl = get_row_col_axes()
    axes_kl = np.array(axes_kl)

    for i, model_name in enumerate(models_in_order):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes_kl[row_idx, col_idx]

        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=18)
        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=18)

        sub_df = results_df[results_df["model_name"] == model_name].copy()
        x = np.arange(len(sub_df))
        labels = [f"Q{i+1}" for i in range(len(sub_df))]
        ax.bar(x, sub_df["kl_divergence"], width=0.5, label="KL", color="brown", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Question")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/kl.png")