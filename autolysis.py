# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "requests>=2.28.0",
#    "pandas>=1.5.0",
#    "matplotlib>=3.5.0",
#    "seaborn>=0.12.0",
#    "numpy>=1.21.0",
#    "rich>=12.0.0",
#    "scipy>=1.9.0",
#    "scikit-learn>=1.0.0",
# ]
# description = "A script for data analysis and visualization."
# entry-point = "autolysis.py"
# ///

import os
import requests
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import shutil
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from functools import wraps

def timeit(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@dataclass
class AnalysisConfig:
    """Configuration settings for analysis."""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")
    time_limit: int = 120  # Updated to 120 seconds

class APIClient:
    """Handles API communication with LLM service."""
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")
        
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def make_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make API request with error handling."""
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            response = requests.post(
                self.proxy_url, 
                headers=self.headers, 
                json=data, 
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return None

class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies."""
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """Create a visualization and save it to a file."""
        pass

class CorrelationHeatmap(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """
        Create a correlation heatmap for numeric columns.
        Optimized for large datasets by using sampling if necessary.
        """
        # Reduce dataset size if needed
        if len(df) > 5000:
            df = df.sample(n=len(df), random_state=42)

        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns available for correlation heatmap.")
            return

        # Create correlation heatmap
        plt.figure(figsize=(10, 6))
        corr_matrix = numeric_df.corr()
        sns.heatmap(
            corr_matrix, 
            annot=False,  # Disable annotations for speed on large datasets
            cmap="coolwarm", 
            center=0
        )
        plt.title(f"Correlation Heatmap - {title}", fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()


class DistributionPlot(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """
        Create distribution plots for numeric columns.
        Optimized for large datasets and includes additional visualizations.
        """
        # Reduce dataset size if needed
        if len(df) > 5000:
            df = df.sample(n=len(df), random_state=42)

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        if n_cols == 0:
            print("No numeric columns available for distribution plots.")
            return

        # Set up grid for multiple plots
        n_rows = (n_cols + 2) // 3
        plt.figure(figsize=(15, 5 * n_rows))

        # Create distribution plots
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, 3, i)
            sns.histplot(df[col], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {col}", fontsize=10)
            plt.xlabel(col)
            plt.ylabel("Frequency")

        # Additional plot: Pairplot for top 3 numeric columns
        if n_cols > 1:
            top_cols = numeric_cols[:min(3, n_cols)]
            pairplot_fig_path = fig_path.with_name(f"{fig_path.stem}_pairplot{fig_path.suffix}")
            sns.pairplot(df[top_cols], corner=True, diag_kind="kde")
            plt.savefig(pairplot_fig_path, dpi=300)
            plt.close()

        plt.suptitle(f"Distribution Analysis - {title}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()


class BoxplotAnalysis(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """
        Create boxplots for numeric columns grouped by categorical columns.
        """
        # Reduce dataset size if needed
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        if numeric_cols.empty or categorical_cols.empty:
            print("Boxplot analysis requires both numeric and categorical columns.")
            return

        # Set up grid for boxplots
        plt.figure(figsize=(15, 6 * len(numeric_cols)))

        for i, num_col in enumerate(numeric_cols, 1):
            for cat_col in categorical_cols[:3]:  # Limit to 3 categorical columns for simplicity
                plt.subplot(len(numeric_cols), 3, i)
                sns.boxplot(x=df[cat_col], y=df[num_col])
                plt.title(f"{num_col} by {cat_col}", fontsize=10)
                plt.xticks(rotation=45)

        plt.suptitle(f"Boxplot Analysis - {title}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()


class TimeSeriesAnalysis(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """
        Create time series plots for datetime columns and numeric data.
        """
        # Reduce dataset size if needed
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)

        datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if datetime_cols.empty or numeric_cols.empty:
            print("Time series analysis requires both datetime and numeric columns.")
            return

        for datetime_col in datetime_cols[:1]:  # Limit to one datetime column
            plt.figure(figsize=(15, 8))
            for num_col in numeric_cols[:5]:  # Limit to 5 numeric columns
                sns.lineplot(x=df[datetime_col], y=df[num_col], label=num_col)

            plt.title(f"Time Series Analysis - {title}", fontsize=16)
            plt.xlabel(datetime_col)
            plt.ylabel("Values")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path.with_name(f"{fig_path.stem}_timeseries{fig_path.suffix}"), dpi=300)
            plt.close()

class LLMAnalyzer:
    """Class to analyze datasets using LLM for code generation and insights."""
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it as an environment variable.")

        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.figure_counter = 0

    def _save_and_close_plot(self, title: str):
        """Save the current plot to a file and close it."""
        self.figure_counter += 1
        filename = f'plot_{self.figure_counter}.png'
        plt.title(title)
        plt.savefig(filename)
        print(f"Plot saved as: {filename}")
        plt.close()

    def _make_llm_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make a request to the LLM API with error handling."""
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            response = requests.post(self.proxy_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {str(e)}")
            return None

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from markdown-formatted text."""
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        return code_blocks if code_blocks else []

    def _execute_code_safely(self, code: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Execute code with safety measures and return success status and error message."""
        try:
            # Ensure the code uses the provided DataFrame
            if 'pd.read_csv' in code:
                raise ValueError("Code should not read from CSV files directly. Use the provided DataFrame 'df'.")

            # Modify the code to save plots instead of showing them
            code = code.replace('plt.show()', 'self._save_and_close_plot("Generated Plot")')

            # Create a restricted locals dictionary with only necessary objects
            local_dict = {
                'pd': pd, 
                'plt': plt, 
                'sns': sns, 
                'df': df, 
                'analyzer': self
            }

            # Execute the code in the restricted environment
            exec(code, {'__builtins__': __builtins__}, local_dict)

            return True, None
        except Exception as e:
            error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            return False, error_msg

    def _fix_code_recursively(self, code: str, error_msg: str, df: pd.DataFrame, max_attempts: int = 3) -> bool:
        """Recursively try to fix code using LLM until it works or max attempts reached."""
        attempt = 0
        while attempt < max_attempts:
            fix_prompt = f"""
            The following Python code generated an error:
            ```python
            {code}
            ```

            Error message:
            {error_msg}

            Please provide a fixed version of the code that:
            1. Handles the error properly
            2. Uses only pandas, matplotlib.pyplot, and seaborn
            3. Works with the DataFrame that has these columns: {df.columns.tolist()}
            4. Includes proper error handling
            5. Uses plt.figure() before creating each plot
            6. Uses plt.show() after each plot is complete
            7. Does not reference any specific CSV files

            Provide ONLY the corrected code block, no explanations.
            """

            messages = [
                {"role": "system", "content": "You are a Python expert focused on data analysis and visualization."},
                {"role": "user", "content": fix_prompt}
            ]

            fixed_content = self._make_llm_request(messages)
            if not fixed_content:
                return False

            fixed_code_blocks = self._extract_code_blocks(fixed_content)
            if not fixed_code_blocks:
                fixed_code = fixed_content  # If no code blocks found, try using the entire response
            else:
                fixed_code = fixed_code_blocks[0]

            success, new_error = self._execute_code_safely(fixed_code, df)
            if success:
                return True

            error_msg = new_error
            attempt += 1

        return False

    def analyze_dataset(self, file_path: str) -> dict:
        """
        Main method to analyze the dataset.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            dict: A dictionary containing the analysis results, issues, and outputs.
        """
        from pathlib import Path
        import pandas as pd
        import traceback

        results = {
            "status": "success",
            "dataset_summary": None,
            "executed_methods": [],
            "code_execution_results": [],
            "issues": []
        }

        try:
            # Validate file path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            if not path.is_file():
                raise ValueError(f"'{file_path}' is not a file.")
            if path.suffix.lower() != '.csv':
                raise ValueError(f"'{file_path}' is not a CSV file.")

            print(f"Loading dataset from: {file_path}")

            # Load and validate dataset with error handling for encoding
            try:
                df = pd.read_csv(file_path, encoding='utf-8')  # Try UTF-8 first
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, trying ISO-8859-1 encoding...")
                df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file is empty.")
            except pd.errors.ParserError:
                raise ValueError("Error parsing the CSV file. Please ensure it's properly formatted.")

            if df.empty:
                raise ValueError("Dataset is empty")

            print(f"Successfully loaded dataset with shape: {df.shape}")

           
            # Get initial analysis suggestions
            data_description = (
                f"Dataset Overview:\n"
                f"Columns: {df.columns.tolist()}\n"
                f"Shape: {df.shape}\n"
                f"Sample data:\n{df.head(3).to_string()}\n"
                f"Data types:\n{df.dtypes.to_string()}"
            )

            initial_prompt = f"""
            Given this dataset description:
            {data_description}

            Generate Python code that:
            1. Calls the predefined analysis methods
            2. Creates meaningful visualizations using matplotlib and seaborn
            3. Calculates relevant summary statistics
            4. Identifies key patterns or relationships
            5. Handles potential errors (missing values, invalid data)
            6. Uses plt.figure() before creating each plot
            7. Uses plt.show() after each plot is complete
            8. Ensures the entire process runs within 2 minutes

            Provide the code in a Python code block.
            """

            messages = [
                {"role": "system", "content": "You are a data scientist specialized in exploratory data analysis."},
                {"role": "user", "content": initial_prompt}
            ]

            # Get and execute initial analysis
            analysis_content = self._make_llm_request(messages)
            if analysis_content:
                code_blocks = self._extract_code_blocks(analysis_content)
                for code in code_blocks:
                    success, error_msg = self._execute_code_safely(code, df)
                    results["code_execution_results"].append({
                        "code": code,
                        "success": success,
                        "error": error_msg if not success else None
                    })
                    if not success:
                        print(f"Initial code execution failed. Attempting to fix...")
                        if not self._fix_code_recursively(code, error_msg, df):
                            results["issues"].append("Failed to fix code after maximum attempts.")
            else:
                results["issues"].append("Failed to generate initial analysis content.")

            # Prepare dataset summary
            results["dataset_summary"] = {
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "sample_data": df.head(3).to_dict(),
                "data_types": df.dtypes.to_dict()
            }

        except Exception as e:
            error_message = f"Analysis failed: {str(e)}"
            results["status"] = "failure"
            results["issues"].append(error_message)
            traceback.print_exc()

        return results

    

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities."""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient()
        self.llm_analyzer = LLMAnalyzer()  # Integrate LLMAnalyzer
        self.visualization_strategies = [
            CorrelationHeatmap(),
            DistributionPlot(),
            BoxplotAnalysis(),
            TimeSeriesAnalysis()
        ]
        self.plots: List[str] = []
        
    @timeit
    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            start_time = time.time()
            self._create_output_directory()
            df = self._load_and_validate_dataset(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Generate visualizations
            self._generate_visualizations(df)

            # Use LLMAnalyzer for analysis
            a = self.llm_analyzer.analyze_dataset(file_path)

            insights = self._generate_insights(df,a)
            
            if time.time() - start_time < self.config.time_limit:
                narrative = self._generate_narrative(df, insights)
                if narrative:
                    self._generate_readme(narrative)
            else:
                print("Time limit reached, skipping narrative generation")
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _create_output_directory(self):
        """Create output directory for saving results."""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset from the given file path."""
        path = Path(file_path)
        if not path.exists() or not path.is_file() or path.suffix.lower() != '.csv':
            raise ValueError(f"Invalid file path: {file_path}")
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
        if df.empty:
            raise ValueError("Dataset is empty")
            
        return df

    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualizations for the dataset."""
        for i, strategy in enumerate(self.visualization_strategies):
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i+1}")
            self.plots.append(viz_path.name)

    def _generate_insights(self, df: pd.DataFrame, a) -> str:
        """Generate insights based on the dataset and statistical analysis."""
        prompt = f"""
        Analyze this dataset based on the following information:

        1. Dataset Statistics:
        {df.describe().to_string()}

        2. Generated Visualizations:
        - {len(self.plots)} plots were generated analyzing different aspects of the data
        3. {a}

        Please provide:
        1. Key patterns and trends
        2. Statistical findings
        3. Notable relationships between variables
        4. Distribution insights
        5. Recommendations

        Format with clear headers and bullet points.
        """

        messages = [
            {"role": "system", "content": "You are a data analyst specializing in statistical analysis."},
            {"role": "user", "content": prompt}
        ]

        insights = self.api_client.make_request(messages)
        if insights:
            print("\nKey Insights Generated")
            return insights
        return ""

    def _generate_narrative(self, df: pd.DataFrame, insights: str) -> str:
        """Generate a narrative based on the dataset analysis."""
        subject = self._determine_subject(df)
        genre = self._determine_genre(df)

        
        story_prompt = f"""
       Craft a heartwarming narrative that unfolds like a contemporary {genre} on {subject}, filled with emotional growth and profound insights (IMPORTANT : REFER AND USE THE FINAL INSIGHTS SECTION THROUGHT THE STORY AND MAKE SURE THAT THE STORY IS CONSISTENT WITH THEM also for every claim made weave in the numbers too also make the process of coming to every conclusion sumer dramatic)

        MAKE THE PROCESS OF ARRIVING TO THESE CONCLUSIONS VERY GRIPPING AND UNIQUE 

        TUG ON EMOTIONS 

        ADD DRAMA ADD LOVE ADD THRILL ADD HERO ENTRY AND COOL SHIT LIKE THAT 

        THE STORY MUST BE VERY MEMEORABLE AND MUST APPEASE INDIAN AUDIENCE BUT YOU CAN MAKE THE STORY NON INDIAN TOO IF NEEDED.

        AGAIN THE FINAL INSIGHTS SECTION IS GODLIKE -- FOLLOW IT AND PRESENT AS MUCH INFO FROM THAT IN THE STORY AS POSSIBLE

        USE {insights}

        Insights:
        {insights}

        Requirements:
        1. Create a compelling story that explains the data journey
        2. Include specific numbers and findings
        3. Make it engaging and memorable
        4. Include implications and recommendations
        5. Use clear sections and structure
        """

        messages = [
            {"role": "system", "content": "You are a data storyteller who transforms analysis into engaging narratives."},
            {"role": "user", "content": story_prompt}
        ]

        narrative = self.api_client.make_request(messages)
        if narrative:
            print("\nNarrative Generated")
            return narrative
        return ""

    def _determine_subject(self, df: pd.DataFrame) -> str:
        """Determine the subject of the dataset based on its content."""
        columns_str = ' '.join(df.columns.str.lower())
        common_subjects = {
            'book': ['book', 'author', 'title', 'publisher'],
            'movie': ['movie', 'film', 'director', 'actor'],
            'sales': ['sale', 'revenue', 'product', 'customer'],
            'financial': ['price', 'cost', 'revenue', 'profit']
        }
        
        for subject, keywords in common_subjects.items():
            if any(keyword in columns_str for keyword in keywords):
                return subject
                
        return "dataset"

   
    def _determine_genre(self, df: pd.DataFrame) -> str:
        """Determine the genre of the dataset based on its content."""
        # Example heuristic: analyze column names for genre-related keywords
        genre_keywords = ['rating', 'review', 'score', 'feedback', 'sentiment']
        for column in df.columns:
            if any(keyword in column.lower() for keyword in genre_keywords):
                return "Analysis of Ratings"  # Example genre
        return "Various Themes"  # Default genre if no keywords found


    def _generate_readme(self, narrative: str):
        """Generate a README file with the analysis narrative and visualizations."""
        readme_content = "# Data Analysis Narrative\n\n"
        readme_content += narrative + "\n\n"
        
        readme_content += "## Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"![{plot}]({plot})\n\n"
            
        readme_path = self.config.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"\nREADME.md generated at: {readme_path}")

def main():
    """Main execution function."""
    # Ensure the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    config = AnalysisConfig(output_dir=Path(Path(file_path).stem))
    analyzer = DataAnalyzer(config)
    analyzer.analyze_dataset(file_path)

if __name__ == "__main__":
    main()
