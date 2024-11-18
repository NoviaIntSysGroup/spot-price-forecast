import os
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from forecaster.utils import get_safe_filename
from forecaster.models.evaluate import calculate_price_extremes, calculate_prediction_accuracy

def save_figure(fig, filename, save_dir):
    """
    Saves a figure to a file.

    Args:
    - fig (matplotlib.figure.Figure): The figure to save.
    - filename (str): The name of the file to save the figure to.
    - save_dir (str): The directory to save the figure to (default is None).
    """
    filename = get_safe_filename(filename)
    if save_dir is not None:
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = filename
    fig.savefig(save_path, bbox_inches='tight')

def plot_time_coefficients(coefficients, col_keywords = ["weekday", "weekend"], title = "", save_dir = ""):
    """
    Plots the coefficients of the linear regression model for weekday or weekend features

    Args:
    - coefficients (pd.Series): Coefficients of the linear regression model.
    - col_keywords (list): List of keywords to filter the coefficients
    - title (str): Title of the plot.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Filter coefficients for weekday or weekend features
    for keyword in col_keywords:
        filtered_coefficients = coefficients[coefficients.index.str.contains(keyword)]
        # Plot the coefficients
        labels = [label.split("_")[-1] for label in filtered_coefficients.index]
        plt.plot(labels, filtered_coefficients, label=keyword)
    plt.xlabel('Hours of Day')
    plt.ylabel('Coefficient Value')
    plt.title(title if title else f'Coefficients for "{col_keywords}" Features')
    plt.xticks(rotation=90)
    plt.legend()
    if save_dir:    
        save_figure(plt, plt.gca().get_title(), save_dir)
    plt.show()


def plot_year_over_year_coefficients(coeffs, keyword=False, model_name="", years=None, save_dir=""):
    """
    Plots the coefficients for each year for the linear regression model.

    Args:
    - coeffs (dict): Dictionary containing the coefficients for each year.
    - keyword (str): Keyword to filter the coefficients.
    - model_name (str): Name of the model.
    - years (list): List of years to plot.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))

    if years:
        coeffs = {year: coeffs[year] for year in years}

    if not coeffs:
        print('No coefficients found for the specified years.')
        return

    for year, coeff in coeffs.items():
        if keyword:
            coeff = coeff[coeff.index.str.contains(keyword)]
            if len(coeff) == 1:
                plt.bar(year, coeff.iloc[0])
            else:
                plt.plot(coeff.index, coeff)
    plt.xlabel('Coefficient Label')
    plt.ylabel('Coefficient Value')
    plt.title(f'{model_name} Coefficients for "{keyword}" Features')
    plt.xticks(rotation=90)  
    # add legend only if its a bar plot
    if len(coeff) != 1 and len(coeffs) > 1:
        plt.legend([f'Year {year}' for year in coeffs.keys()], title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

    # add year in title if only one year is selected
    if len(coeffs) == 1:
        plt.title(f'{model_name} Coefficients for {keyword} Features in {year}')  
    
    if save_dir:
        save_figure(plt, plt.gca().get_title(), save_dir)
    plt.show()

def plot_prediction_accuracy_histogram(accuracy_dict, title, year_on_year=False, save_dir=""):
    """
    Plots separate bar graphs for max and min price hour prediction accuracies in the same image.
    Displays the accuracy values above each bar.
    
    Parameters:
    - accuracy_dict: Dictionary with keys as k values and values as accuracy percentages for max and min hours
        - Key: 'max' or 'min'
        - Value: Dictionary with keys as k values and values as accuracy percentages
    - title: Title for the entire figure
    - year_on_year: Boolean indicating if the data is for each year separately
    - save_dir: Directory to save the plot
    """
    if year_on_year:
        years = sorted(accuracy_dict['max'].keys())
        x = list(accuracy_dict['max'][years[0]].keys())  # Assuming all years have the same k values

        # Setup the plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Adjust bar width and spacing
        bar_width = 0.15  # Smaller bar width for better spacing
        space_between_groups = 0.3  # Space between groups of bars
        positions = [i * (bar_width * len(years) + space_between_groups) for i in range(len(x))]

        # Plot Max and Min Price Hours Accuracy for each year
        for i, (key, ax, label) in enumerate(zip(['max', 'min'], axes, ['Max', 'Min'])):
            for j, year in enumerate(years):
                # Get the accuracy values for the current year
                y_values = list(accuracy_dict[key][year].values())
                # Adjust positions for each year bar
                ax.bar(
                    [pos + j * bar_width for pos in positions], 
                    y_values, 
                    width=bar_width, 
                    label=f'Year {year}'
                )

            # Set labels and titles
            ax.set_xlabel('Number of Top-k Hours')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{label} Price Hours Accuracy')
            ax.set_xticks([pos + bar_width * (len(years) - 1) / 2 for pos in positions])
            ax.set_xticklabels(x)
            ax.set_ylim(0, 100)

            # Add value labels on top of each bar
            for j, year in enumerate(years):
                y_values = list(accuracy_dict[key][year].values())
                for k, v in enumerate(y_values):
                    ax.text(positions[k] + j * bar_width, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)
            
            # Add legend
            ax.legend()

        fig.suptitle(title)
        plt.tight_layout()
        if save_dir:
            save_figure(fig, title, save_dir)
        plt.show()
    else:
        x = list(accuracy_dict['max'].keys())
        y_max = list(accuracy_dict['max'].values())
        y_min = list(accuracy_dict['min'].values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot Max and Min Price Hours Accuracy
        for ax, y, label, color in zip(axes, [y_max, y_min], ['Max', 'Min'], ['skyblue', 'lightcoral']):
            ax.bar(x, y, color=color)
            ax.set_xlabel('Number of Top-k Hours')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{label} Price Hours Accuracy')
            ax.set_xticks(x)
            ax.set_ylim(0, 100)
            for i, v in enumerate(y):
                ax.text(i + 1, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)

        fig.suptitle(title)
        plt.tight_layout()

        if save_dir:
            save_figure(fig, title, save_dir)
        plt.show()

def plot_overall_accuracy_comparison(accuracy_dicts, models, top_k_values, add_to_title, save_dir=""):
    """
    Plots a multi-bar chart comparing overall accuracies across different models for different top-k values.

    Parameters:
    - accuracy_dicts: List of accuracy dictionaries for each model
    - models: List of model names
    - top_k_values: List of top-k values considered
    - add_to_title: Additional text to add to the title
    - save_dir: Directory to save the plot
    """
    x = top_k_values
    bar_width = 0.2
    space_between_groups = 0.3  # Space between groups of bars
    positions = [i * (bar_width * len(models) + space_between_groups) for i in range(len(x))]

    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Max and Min Price Hours Accuracy for each model
    for _, (key, ax, label) in enumerate(zip(['max', 'min'], axes, ['Max', 'Min'])):
        for j, model in enumerate(models):
            # Get the accuracy values for the current model
            y_values = [accuracy_dicts[model][key][k] for k in x]
            # Adjust positions for each model bar
            ax.bar(
                [pos + j * bar_width for pos in positions], 
                y_values, 
                width=bar_width, 
                label=f'Model: {model}'
            )

        # Set labels and titles
        ax.set_xlabel('Number of Top-k Hours')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{label} Price Hours Accuracy')
        ax.set_xticks([pos + bar_width * (len(models) - 1) / 2 for pos in positions])
        ax.set_xticklabels(x)
        ax.set_ylim(0, 100)

        # Add value labels on top of each bar
        for j, model in enumerate(models):
            y_values = [accuracy_dicts[model][key][k] for k in x]
            for k, v in enumerate(y_values):
                ax.text(positions[k] + j * bar_width, v + 2, f'{v:.1f}%', ha='center', fontsize=10, rotation=90)
        
        # Add legend
        ax.legend()
    sup_title = f'Overall Prediction Accuracy Comparison Across Models {add_to_title}'
    fig.suptitle(sup_title)
    plt.tight_layout()

    if save_dir:
        save_figure(fig, sup_title, save_dir)

    plt.show()

def plot_custom_metrics(spot_data, predictions, add_to_title="", top_k=3, save_dir=""):
    """
    Plot custom metrics for the models

    Args:
    spot_data: pd.DataFrame, spot price data
    predictions: dict, predictions
    add_to_title: str, additional string to add to the title
    top_k: int, number of top-k hours to consider
    save_dir: str, directory to save the plot
    """    

    actual_extremes = calculate_price_extremes(pd.DataFrame(spot_data['y'][spot_data.index.year != 2016], columns=['y']), price_column='y', top_k=top_k)

    # Initialize a dictionary to store overall accuracies
    overall_accuracies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Calculate the accuracy of the predictions for each model
    for model, prediction in predictions.items():
        for split_set, values  in prediction.items():
            predicted_extremes = calculate_price_extremes(pd.DataFrame(values, columns=['y_hat']), price_column='y_hat', top_k=top_k)
            
            # Calculate year-wise accuracy
            accuracy_dict_yearwise = calculate_prediction_accuracy(actual_extremes, predicted_extremes, order=True, top_k=top_k, year_on_year=True)
            title = f"Accuracy of Top-k Hour Predictions [{model}][{split_set.capitalize()}]{add_to_title}"
            plot_prediction_accuracy_histogram(accuracy_dict_yearwise, title=title, year_on_year=True, save_dir=save_dir)
            
            # Calculate overall accuracy
            accuracy_dict_overall = calculate_prediction_accuracy(actual_extremes, predicted_extremes, order=True, top_k=top_k, year_on_year=False)
            overall_accuracies[split_set][model] = accuracy_dict_overall

    # Plot the overall accuracy comparison across models
    models = list(predictions.keys())

    for split_set in overall_accuracies.keys():
        plot_overall_accuracy_comparison(
            overall_accuracies[split_set], 
            models, 
            top_k_values=range(1, top_k + 1), 
            add_to_title=f"[{split_set.capitalize()}]", 
            save_dir=save_dir
        )
