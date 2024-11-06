import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_project_root():
    """
    It returns the path to the project root

    :return: The path to the root of the project.
    """
    utils_path = os.path.dirname(os.path.abspath(__file__))
    root_end_idx = list(re.finditer("src", utils_path))[-1].start()
    root_path = utils_path[0:root_end_idx]
    return root_path


############################


def calculate_price_extremes(df, price_column='y', date_column=None, top_k=1):
    """
    Calculates the top-k hours with the maximum and minimum prices for each day in the dataset.
    Works with both a datetime index or a separate date column.

    Parameters:
    - df: pandas DataFrame containing the data
    - price_column: str, name of the column containing the price data (default is 'y')
    - date_column: str, name of the column containing the date data, if not using the index (default is None)
    - top_k: int, number of top maximum and minimum price hours to return (default is 1)

    Returns:
    - daily_extremes: pandas DataFrame with columns 'day', 'max_price_hours', 'min_price_hours', and 'year'
    """
    df_copy = df.copy()
    
    # Handle datetime conversion
    if date_column:
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy['day'] = df_copy[date_column].dt.date
        df_copy['hour'] = df_copy[date_column].dt.hour
    else:
        df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['day'] = df_copy.index.date
        df_copy['hour'] = df_copy.index.hour
    
    # Extract year
    df_copy['year'] = pd.to_datetime(df_copy['day']).dt.year
    
    # Sort data to allow easy nlargest/nsmallest calculation
    df_copy.sort_values(by=['day', price_column], ascending=[True, False], inplace=True)
    
    # Get top-k max prices per day
    max_price_hours = df_copy.groupby('day').head(top_k).groupby('day')['hour'].apply(list)
    
    # Sort data to get min prices
    df_copy.sort_values(by=['day', price_column], ascending=[True, True], inplace=True)
    
    # Get top-k min prices per day
    min_price_hours = df_copy.groupby('day').head(top_k).groupby('day')['hour'].apply(list)
    
    # Combine results into a DataFrame
    daily_extremes = pd.DataFrame({
        'day': max_price_hours.index,
        'max_price_hours': max_price_hours.values,
        'min_price_hours': min_price_hours.reindex(max_price_hours.index).values,
        'year': pd.to_datetime(max_price_hours.index).year
    })

    return daily_extremes.reset_index(drop=True)


def calculate_prediction_accuracy(actual_extremes, predicted_extremes, year_on_year=False, order=False, top_k=1):
    """
    Calculates the percentage of correct predictions for top-k hours, either year-by-year or overall.

    Args:
    - actual_extremes (pd.DataFrame): DataFrame containing the actual extreme hours.
    - predicted_extremes (pd.DataFrame): DataFrame containing the predicted extreme hours.
    - year_on_year (bool): If True, calculate accuracy separately for each year. If False, calculate overall accuracy (default is False).
    - order (bool): If True, the order of the hours must match exactly (default is False).
    - top_k (int): Number of top hours to consider (default is 1).

    Returns:
    - accuracy_dict (dict): Dictionary containing the accuracy percentages for max and min hours.
        - Key: 'max' or 'min'
        - Value: Dictionary with keys as k values and values as accuracy percentages
    """
    accuracy_dict = {'max': {}, 'min': {}}
    
    # Merge actual and predicted data on 'day'
    merged_df = pd.merge(actual_extremes, predicted_extremes, on='day', suffixes=('_actual', '_predicted'))
    
    # Group by year if year_on_year is True
    if year_on_year:
        grouped = merged_df.groupby('year_actual')
    else:
        grouped = [(None, merged_df)]
    
    # Calculate accuracy for each group
    for year, group in grouped:
        # Calculate accuracy for each k value
        for k in range(1, top_k + 1):
            # Calculate accuracy for max and min hours
            for key in ['max', 'min']:
                # Get the actual and predicted columns
                actual_col = f'{key}_price_hours_actual'
                predicted_col = f'{key}_price_hours_predicted'
                
                if order:
                    # Compare the lists directly for ordered comparison
                    correct_predictions = (group[actual_col].str[:k] == group[predicted_col].str[:k]).sum()
                else:
                    # Convert to sets for unordered comparison
                    correct_predictions = (group[actual_col].str[:k].apply(set) == group[predicted_col].str[:k].apply(set)).sum()
                
                if year_on_year:
                    if year not in accuracy_dict[key]:
                        accuracy_dict[key][year] = {}
                    accuracy_dict[key][year][k] = (correct_predictions / len(group)) * 100
                else:
                    accuracy_dict[key][k] = (correct_predictions / len(group)) * 100

    return accuracy_dict


def plot_prediction_accuracy_histogram(accuracy_dict, title, year_on_year=False):
    """
    Plots separate bar graphs for max and min price hour prediction accuracies in the same image.
    Displays the accuracy values above each bar.
    
    Parameters:
    - accuracy_dict: Dictionary with keys as k values and values as accuracy percentages for max and min hours
        - Key: 'max' or 'min'
        - Value: Dictionary with keys as k values and values as accuracy percentages
    - title: Title for the entire figure
    - year_on_year: Boolean indicating if the data is for each year separately
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
        plt.show()

def plot_overall_accuracy_comparison(accuracy_dicts, models, top_k_values, add_to_title):
    """
    Plots a multi-bar chart comparing overall accuracies across different models for different top-k values.

    Parameters:
    - accuracy_dicts: List of accuracy dictionaries for each model
    - models: List of model names
    - top_k_values: List of top-k values considered
    - add_to_title: Additional text to add to the title
    """
    x = top_k_values
    bar_width = 0.2
    space_between_groups = 0.3  # Space between groups of bars
    positions = [i * (bar_width * len(models) + space_between_groups) for i in range(len(x))]

    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Max and Min Price Hours Accuracy for each model
    for i, (key, ax, label) in enumerate(zip(['max', 'min'], axes, ['Max', 'Min'])):
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

    fig.suptitle(f'Overall Prediction Accuracy Comparison Across Models {add_to_title}')
    plt.tight_layout()
    plt.show()