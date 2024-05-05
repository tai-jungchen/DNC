"""
file name: plot_generator.py
Author: Alex

Generates all the plots needed for the paper.
"""
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Call out the visualization functions.
    """
    # global_vs_local_bar_chart()
    class_dist_chart("binary")
    class_dist_chart("multi")


def class_dist_chart(label):
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    if label == "binary":
        df["target"].value_counts().plot.bar(rot=0)
        plt.ylabel("Count")
        plt.xlabel("Binary Label")
    elif label == "multi":
        df["failure.type"].value_counts().plot.bar(rot=0)
        plt.ylabel("Count")
        plt.xlabel("Multi-class Label")
    else:
        raise Exception("Invalid label type.")
    plt.show()


def global_vs_local_bar_chart():
    """
    Plot out the bar chart to compare the performance of global and local models on different sub data.
    """
    # model and metric to be plotted #

    # model = "Logistic Regression"
    # model = "Decision Tree"
    # model = "Random Forest"
    model = "XGBoost"

    # metric = "Balance Accuracy"
    # metric = "Precision"
    # metric = "Recall"
    metric = "F1 score"

    # Sample data (accuracy values for local and global models for five datasets)
    data = {
        'Sub-Data': ['Sub-Data 1', 'Sub-Data 2', 'Sub-Data 3', 'Sub-Data 4', 'Sub-Data 5'],
        'Global Model': [0.7737, 0.6506, 0.5301, 0.6791, 0.0728],
        'Local Model': [0.9481, 0.8167, 0.8347, 0.8504, 0.0304]
    }

    # Convert data to Pandas DataFrame
    df = pd.DataFrame(data)

    # Set the width of the bars
    bar_width = 0.35

    # Set the position of the bars on the x-axis
    r1 = range(len(df['Sub-Data']))
    r2 = [x + bar_width for x in r1]

    # Plotting the bars
    plt.bar(r1, df['Local Model'], color='blue', width=bar_width, edgecolor='grey', label='Local Model')
    plt.bar(r2, df['Global Model'], color='orange', width=bar_width, edgecolor='grey', label='Global Model')

    # Adding labels
    plt.xlabel('Sub-Data', fontweight='bold')
    plt.ylabel(metric, fontweight='bold')
    plt.title('Comparison of Local and Global Models by Sub-Data on ' + metric, fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(df['Sub-Data']))], df['Sub-Data'])

    # value on plot
    for i in r1:
        plt.text(i, df['Local Model'][i] + 0.01, f"{df['Local Model'][i]:.4f}", ha='center', va='bottom')
        plt.text(i + bar_width, df['Global Model'][i] + 0.01, f"{df['Global Model'][i]:.4f}", ha='center', va='bottom')

    # Adding a legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
