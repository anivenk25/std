
# Required libraries are given in meta data as requires and dependencies. So no need to install each time in pip command

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "chardet>=5.2.0",
#   "matplotlib>=3.9.3",
#   "numpy>=2.2.0",
#   "openai>=1.57.2",
#   "pandas>=2.2.3",
#   "python-dotenv>=1.0.1",
#   "requests>=2.32.3",
#   "scikit-learn>=1.6.0",
#   "seaborn>=0.13.2",
# ]
# ///

iimport chardet
import base64
from dotenv import load_dotenv
import warnings
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from functools import lru_cache


"""
 load_dotenv() used to load environment variables from a .env file into the environment.
 To manage configuration settings, especially  to avoid hardcoding sensitive information (e.g., API keys, database credentials) directly in the code."""

load_dotenv()

# SET AIPROXY_TOKEN
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("ERROR: AIPROXY_TOKEN environment variable is not set.")


API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

"""@lru_cache: This decorator comes from the functools module. It automatically stores the results of function calls in a cache. The next time the function is called with the same arguments, the cached result is returned instead of recalculating it.
maxsize=10: This parameter sets the maximum number of cached results to keep. Once the cache exceeds this size, the least recently used results are discarded. If you set maxsize=None, the cache size becomes unlimited, but with a specific value (like 10), the cache will only store 10 entries."""

@lru_cache(maxsize=10)


def query_chat_completion(prompt, model="gpt-4o-mini"):
    """Sending a chat prompt to the LLM and cache results to optimize API interactions."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content returned.")
    except requests.RequestException as e:
        raise Exception(f"Error during LLM query: {e}")


def detect_file_encoding(filepath):
    """chardet is a Python library used to detect the character encoding of text data.
    Useful when you're working with files or data from sources where the encoding isn't specified, and we need to determine the correct encoding to properly read or write the data."""
    #Detecting the encoding of a file.
    with open(filepath, "rb") as file:
        result = chardet.detect(file.read(100000))
        return result["encoding"]

def load_data(filename):
    """Loading CSV data into a Pandas DataFrame, handling file encoding with fallbacks."""
    try:
        encoding = detect_file_encoding(filename)
        print(f"Detected encoding for {filename}: {encoding}")

        return pd.read_csv(filename, encoding=encoding)
    except Exception as primary_error:
        print(f"Primary encoding {encoding} failed: {primary_error}")

        fallback_encodings = ["utf-8-sig", "latin1"]
        for fallback in fallback_encodings:
            try:
                print(f"Trying fallback encoding: {fallback}")
                return pd.read_csv(filename, encoding=fallback)
            except Exception as fallback_error:
                print(f"Fallback encoding {fallback} failed: {fallback_error}")

        raise ValueError(f"Could not to load file {filename} with any encoding.")

""" CREATE OUTPUT FOLDER"""
def create_output_folder(filename):
    print("Output folder is created")
    folder_name = os.path.splitext(os.path.basename(filename))[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

#GENERAL ANALYSIS LIKE MISSING VALUES,
def generic_analysis_data(df):
    print("Calling genereic analysis")
    """Performing generic analysis on the dataset."""
    analysis = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict(),
        "variance": df.var(numeric_only=True).to_dict(),
        "skewness": df.skew(numeric_only=True).to_dict()
    }
    return analysis

def preprocessing_data(df):
    print("Preprocessing data")
    """Preprocess data to handle missing values."""
    numeric_df = df.select_dtypes(include=['float', 'int'])
    imputer = SimpleImputer(strategy='mean')
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    return numeric_df_imputed

def preprocess_for_visualization(df, max_rows=1000):
    print("Getting ready for visualizations")
    """Limiting the dataset to a subset for faster visualizations."""
    if df.shape[0] > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

def detecting_features(df):
    print("Detecting features")
    """Detect the feature types for special analyses."""
    return {
        "time_series": df.select_dtypes(include=['datetime']).columns.tolist(),
        "geographic": [col for col in df.columns if any(geo in col.lower() for geo in ["latitude", "longitude", "region", "country"])],
        "network": [col for col in df.columns if "source" in col.lower() or "target" in col.lower()],
        "cluster": df.select_dtypes(include=['float', 'int']).columns.tolist()
    }

def perform_specific_analyses(df, feature_types):
    """Perform specfic analyses based on feature types."""
    print("Performing specific analysis")
    analyses = {}

    if feature_types["time_series"]:
        analyses["time_series"] = [
            f"Time-series features detected: {', '.join(feature_types['time_series'])}. "
            "These can be used to observe trends or forecast future patterns."
        ]
    else:
        analyses["time_series"] = ["No time-series features detected."]

    if len(feature_types["geographic"]) >= 2:
        analyses["geographic"] = [
            f"Geographic features detected: {', '.join(feature_types['geographic'][:2])}. "
            "These can be used to visualize or analyze spatial distributions."
        ]
    else:
        analyses["geographic"] = ["No geographic features detected."]

    if len(feature_types["network"]) >= 2:
        analyses["network"] = [
            f"Network relationships detected between {feature_types['network'][0]} and {feature_types['network'][1]}. "
            "These can be analyzed for connectivity or collaborations."
        ]
    else:
        analyses["network"] = ["No network features detected."]

    if len(feature_types["cluster"]) > 1:
        analyses["cluster"] = [
            "Cluster analysis is feasible with the available numeric features. "
            "This could help identify natural groupings in the data."
        ]
    else:
        analyses["cluster"] = ["Not enough numeric features for cluster analysis."]

    return analyses

def advanced_analysis(df):
    """Performing advanced statistical and exploratory data analysis."""
    print("Performing advanced statistical analysiis")
    analysis = {}

    correlation_matrix = df.corr()
    high_corr_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
    significant_corr = high_corr_pairs[high_corr_pairs > 0.7].drop_duplicates()
    analysis["significant_correlations"] = significant_corr.to_dict()

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 1:
        from scipy.stats import chi2_contingency
        chi_results = {}
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                contingency_table = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi_results[f"{categorical_cols[i]} vs {categorical_cols[j]}"] = {"chi2": chi2, "p_value": p}
        analysis["chi_square_tests"] = chi_results

    if len(df.select_dtypes(include=['float', 'int']).columns) > 1:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df.select_dtypes(include=['float', 'int']))
        analysis["kmeans_clusters"] = pd.Series(kmeans.labels_).value_counts().to_dict()

    return analysis

def generate_prompt(data_summary, feature_types):
    """Generating dynamic prompts based on dataset properties."""
    print("Genrating prompt")
    if len(data_summary["columns"]) > 50:
        return "The dataset has many columns. Focus on identifying the most critical features and summarizing insights concisely."
    elif "time_series" in feature_types and feature_types["time_series"]:
        return "The dataset contains time-series data. Provide detailed temporal trends and predictions."
    else:
        return "Analyze the dataset comprehensively and highlight correlations, distributions, and any anomalies."


def agentic_workflow(data_summary, feature_types):
    """Performing iterative multi-step analysis based on LLM responses."""
    print("Multiple step analysis")
    prompt = generate_prompt(data_summary, feature_types)
    initial_insights = query_chat_completion(prompt)

    if "missing values" in initial_insights.lower():
        refinement_prompt = "You mentioned missing values. Suggest specific imputation strategies based on data types."
        refinement = query_chat_completion(refinement_prompt)
        return initial_insights + "\n" + refinement
    else:
        return initial_insights

def create_visualizations(df, output_folder):
    numeric_df = preprocess_data(df)
    visualization_df = preprocess_for_visualization(numeric_df)

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", cbar_kws={'shrink': 0.8})
        plt.title("Correlation Heatmap", fontsize=16)
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()

    if visualization_df.shape[1] > 1:
        model = IsolationForest(random_state=42)
        visualization_df['outlier_score'] = model.fit_predict(visualization_df)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=visualization_df, x=visualization_df.columns[0], y=visualization_df.columns[1], hue='outlier_score', palette="Set1")
        plt.title("Outlier Detection (Scatter Plot)", fontsize=16)
        plt.xlabel(visualization_df.columns[0])
        plt.ylabel(visualization_df.columns[1])
        plt.legend(title="Outliers")
        plt.savefig(os.path.join(output_folder, "outlier_detection.png"))
        plt.close()

    if visualization_df.shape[1] > 1:
        selected_columns = visualization_df.columns[:5]
        sns.pairplot(visualization_df[selected_columns], palette="husl")
        plt.savefig(os.path.join(output_folder, "pairplot_analysis.png"))
        plt.close()

    return [
        os.path.join(output_folder, "correlation_heatmap.png"),
        os.path.join(output_folder, "outlier_detection.png"),
        os.path.join(output_folder, "pairplot_analysis.png")
    ]


def image_to_base64(image_path):
    """ Images are converted into base 64. Base64 encoding is used for encoding binary data (such as images, files, or other non-text data) into a text format.
    This encoding is useful in situations where binary data needs to be safely transmitted or stored in environments that primarily handle text."""
    print("Converting images into base64")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_image_with_vision_api(image_path, model="gpt-4o-mini"):
    
    """Analyze an image using the OpenAI Vision API."""
    print("Image analysis")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Analyze these images in an interesting manner."}],
    "image": image_to_base64(image_path),
    }
    response = requests.post(API_URL, headers=headers, json=payload)

def llm_narrate_story(summary, insights, advanced_analyses, charts, special_analyses):
    """Generate a cohesive and structured narrative story."""
    print("Narrating summary")
    special_analyses_summary = "\n".join(
        f"{key.capitalize()} Analysis:\n" + "\n".join(value)
        for key, value in special_analyses.items()
    )
    advanced_analyses_summary = "\n".join(
        f"{key.capitalize()} Findings:\n{value}" for key, value in advanced_analyses.items()
    )
    prompt = (
        "As a creative story teller generate a cohesive and structured narrative story.
        f"The dataset has the following properties:\n{summary}\n"
        f"Insights:\n{insights}\n"
        f"Advanced Analysis:\n{advanced_analyses_summary}\n"
        f"Special Analyses:\n{special_analyses_summary}\n"
        f"The visualizations generated are: {', '.join(charts)}.\n"
        "Please generate a well-structured Markdown report covering data properties, analysis, insights, visualizations, and implications. "
        "Ensure that the content flows logically and highlights key findings with proper emphasis. "
        "Use headings, bullet points, and descriptions to enhance readability."
    )
    return query_chat_completion(prompt)


"""def save_readme(content):
    #Save README file
     with open('README.md', 'w', encoding='utf-8') as file:
        file.write(content) 
     print(f"README.md saved")"""

def save_readme(content, output_folder):
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(content)
    print(f"README.md saved at {readme_path}")



if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset = sys.argv[1]

    # Generate output folder name based on dataset name (without extension)
    output_folder = os.path.splitext(os.path.basename(dataset))[0]
    # Suppress the specific warning
    warnings.filterwarnings("ignore", message="Ignoring `palette` because no `hue` variable has been assigned")
    # Create the output folder dynamically
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")

    try:
        df = load_data(dataset)

        summary = generic_analysis_data(df)
        print("Generic analysis completed.")
        
        df = preprocessing_data(df)

        advanced_analyses = advanced_analysis(df)
        print("Advanced analysis completed.")

        features = detecting_features(df)

        specific_analyses = perform_specific_analyses(df, features)

        insights = agentic_workflow(summary, features)
        print("LLM insights generated.")

        charts = generate_visualizations(df, output_folder)  # Save visualizations in the output folder
        print("Visualizations Generated.")

        story = llm_narrate_story(summary, insights, advanced_analyses, charts, specific_analyses)
        print("Narrative Done.")

        #save_readme(story)  # Save README.md in the output folder
        
        save_readme(story, output_folder)  # Save README.md in the output folder
        print(f"README.md generated in {output_folder}.")
    except Exception as e:
        print("Error:", e)
