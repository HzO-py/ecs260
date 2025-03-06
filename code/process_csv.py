import pandas as pd
import matplotlib.pyplot as plt

def load_schema(schema_path):
    """Load schema and create a mapping of column names to their type."""
    schema_df = pd.read_csv(schema_path, encoding="utf-8", low_memory=False)
    
    # Create a dictionary mapping qname to selector type
    schema_dict = schema_df.set_index("qname")["selector"].to_dict()
    
    return schema_dict

def is_numeric_column(data):
    """Check if all values in a column are numeric."""
    numeric_data = pd.to_numeric(data, errors="coerce").dropna()
    return len(numeric_data) == len(data)  # If all values are numeric

def process_column(public_df, column_name, column_type):
    """Process and plot the distribution of a column based on its type."""
    if column_name not in public_df.columns:
        print(f"Column '{column_name}' not found in public dataset.")
        return

    data = public_df[column_name].dropna()

    if is_numeric_column(data):
        column_type = "SL"
        answer_type='number'
    else:
        answer_type='multiple options' if column_type == "MAVR" else 'single option'

    print(f"Processing column: {column_name} (Type: {answer_type})")


    if column_type in ["SAVR", "DL"]:  # Single choice
        distribution = data.value_counts(sort=False)

    elif column_type == "MAVR":  # Multiple choice (split by ";")
        all_choices = data.str.split(";").explode()
        distribution = all_choices.value_counts(sort=False)

    elif column_type in ["SL", "NPS", "HSLIDER"]:  # Numeric values
        data = pd.to_numeric(data, errors="coerce").dropna()

        # lower_bound, upper_bound = data.quantile(0.01), data.quantile(0.99)
        # data = data.clip(lower=lower_bound, upper=upper_bound)

        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Distribution of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
        return

    else:
        print(f"Unknown column type '{column_type}' for column '{column_name}'.")
        return

    # Plot categorical distribution
    plt.figure(figsize=(12, 6))
    distribution.plot(kind="bar", color="skyblue")
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()  # 非阻塞显示

def analyze_column(schema_path, public_path, column_name):
    """Main function to analyze a specific column based on schema."""
    schema_dict = load_schema(schema_path)
    column_type = schema_dict.get(column_name, None)

    if not column_type:
        print(f"Column '{column_name}' not found in schema.")
        return

    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    process_column(public_df, column_name, column_type)


# Example usage
schema_path = "./raw_data/survey_results_schema.csv"  # Replace with your schema file path
public_path = "./raw_data/survey_results_public.csv"  # Replace with your public data file path





def analyze_all_columns(schema_path, public_path):
    """Analyze all columns in public.csv based on schema."""
    schema_dict = load_schema(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)

    for column_name in public_df.columns:
        column_type = schema_dict.get(column_name, None)
        if column_type:
            process_column(public_df, column_name, column_type)


# 如果你只想看survey_results_public.csv第一行某个指定的问题column_name
# column_name = "LearnCodeOnline"  # Replace with the column you want to analyze
# analyze_column(schema_path, public_path, column_name)
# exit(0)


# 如果你想看所有问题
# analyze_all_columns(schema_path, public_path)

import pandas as pd
import matplotlib.pyplot as plt

def filter_savr_qnames(schema_path):
    """Extract qnames where selector is 'SAVR' from schema file (starting from row 18)."""
    schema_df = pd.read_csv(schema_path, encoding="utf-8", low_memory=False)
    
    # Skip first 17 rows
    schema_df = schema_df.iloc[17:]
    
    # Filter rows where 'selector' column is 'SAVR'
    savr_qnames = schema_df[schema_df["selector"] == "SAVR"]["qname"].tolist()
    
    return savr_qnames

def plot_savr_columns(schema_path, public_path):
    """Plot and count non-NA responses for SAVR qnames in public data."""
    savr_qnames = filter_savr_qnames(schema_path)
    public_df = pd.read_csv(public_path, encoding="utf-8", low_memory=False)
    
    for qname in savr_qnames:
        if qname in public_df.columns:
            data = public_df[qname].dropna()
            response_count = len(data)
            
            print(f"Column: {qname}, Non-NA Responses: {response_count}")
            
            plt.figure(figsize=(12, 6))
            data.value_counts().sort_index(key=lambda x: x.astype(str)).plot(kind="bar", color="skyblue")
            plt.title(f"Distribution of {qname}")
            plt.xlabel(qname)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

# Example usage
schema_path = "survey_results_schema.csv"  # Replace with your schema file path
public_path = "survey_results_public.csv"  # Replace with your public data file path
plot_savr_columns(schema_path, public_path)

