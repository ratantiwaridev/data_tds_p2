import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import openai
import numpy as np
from sklearn.cluster import KMeans

# Check if AI Proxy token is set
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

token = os.environ["AIPROXY_TOKEN"]
openai.api_key = token

def analyze_csv(file_path):
    """
    Analyze the CSV file and return insights and data for visualizations.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Basic analysis
    summary = data.describe(include='all').to_string()
    missing_values = data.isnull().sum().to_dict()
    data_types = data.dtypes.to_dict()
    column_names = data.columns.tolist()

    # Correlation matrix for numerical data
    correlation_matrix = None
    if data.select_dtypes(include=[np.number]).shape[1] > 1:
        correlation_matrix = data.select_dtypes(include=[np.number]).corr()

    # Generate a prompt for AI
    prompt = (
        f"You are an expert data analyst. The dataset contains the following columns: {', '.join(column_names)}. "
        f"Here is the summary of the dataset: \n{summary}\n "
        f"Missing values: {missing_values}\nData types: {data_types}\n"
    )

    if correlation_matrix is not None:
        prompt += f"Here is the correlation matrix for numerical columns: \n{correlation_matrix.to_string()}\n"

    prompt += "Please write an engaging narrative analysis of the dataset and suggest 1-3 visualizations that would best represent its key insights."

    # Use OpenAI's GPT-4o-Mini model to generate a narrative
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )
        narrative = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with AI Proxy: {e}")
        sys.exit(1)

    return data, narrative, correlation_matrix

def generate_visualizations(data, correlation_matrix, output_prefix):
    """
    Generate visualizations based on the dataset and save them as PNG files.
    """
    visualizations = []

    # Example visualization: Bar chart of the first column's value counts (if applicable)
    if len(data.columns) >= 2:
        col = data.columns[0]
        counts = data[col].value_counts().head(10)

        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar', color='skyblue')
        plt.title(f"Top 10 {col} Counts")
        plt.xlabel(col)
        plt.ylabel("Count")
        output_path = f"{output_prefix}_barchart.png"
        plt.savefig(output_path)
        plt.close()

        visualizations.append(output_path)

    # Correlation heatmap
    if correlation_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix Heatmap")
        output_path = f"{output_prefix}_heatmap.png"
        plt.savefig(output_path)
        plt.close()

        visualizations.append(output_path)

    # Outlier detection (boxplot for numerical columns)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[col], color="lightblue")
            plt.title(f"Boxplot for {col}")
            output_path = f"{output_prefix}_{col}_boxplot.png"
            plt.savefig(output_path)
            plt.close()

            visualizations.append(output_path)

    return visualizations

def create_readme(narrative, visualizations):
    """
    Create a README.md file summarizing the analysis and embedding visualizations.
    """
    template = Template(
        """
        # Automated Analysis Report

        {{ narrative }}

        {% for vis in visualizations %}
        ![Visualization](./{{ vis }})
        {% endfor %}
        """
    )

    readme_content = template.render(narrative=narrative, visualizations=visualizations)
    with open("README.md", "w") as f:
        f.write(readme_content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Analyze the dataset
    data, narrative, correlation_matrix = analyze_csv(csv_file)

    # Generate visualizations
    output_prefix = os.path.splitext(os.path.basename(csv_file))[0]
    visualizations = generate_visualizations(data, correlation_matrix, output_prefix)

    # Create README.md
    create_readme(narrative, visualizations)

    print("Analysis complete. Results saved to README.md and visualization files.")

if __name__ == "__main__":
    main()
