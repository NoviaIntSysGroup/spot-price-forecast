import pandas as pd
import matplotlib.pyplot as plt


def plot_coefficients(coefficients, col_keyword):
    """
    Plots the coefficients of the linear regression model for weekday or weekend features

    Args:
    - coefficients (pd.Series): Coefficients of the linear regression model.
    - weekend (bool): Whether to plot coefficients for weekend features.

    Returns:
    - None
    """
    # Filter coefficients for weekday or weekend features
    if col_keyword:
        coefficients = coefficients[coefficients.index.str.contains(col_keyword)]
    
    # Plot the coefficients
    plt.figure(figsize=(10, 6))
    plt.plot(coefficients.index, coefficients)
    plt.xlabel('Coefficient Value')
    plt.title(f'Coefficients for {col_keyword} features')
    plt.xticks(rotation=90)
    plt.show()

def plot_year_over_year_coefficients(coeffs, keyword=False, model_name="", years=None):
    """
    Plots the coefficients for each year for the linear regression model.

    Args:
    - coeffs (dict): Dictionary containing the coefficients for each year.
    - keyword (str): Keyword to filter the coefficients.
    - model_name (str): Name of the model.
    - years (list): List of years to plot.

    Returns:
    - None
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
    plt.xlabel('Coefficient Value')
    plt.title(f'{model_name} Coefficients for "{keyword}" Features')
    plt.xticks(rotation=90)  
    # add legend only if its a bar plot
    if len(coeff) != 1 and len(coeffs) > 1:
        plt.legend([f'Year {year}' for year in coeffs.keys()], title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

    # add year in title if only one year is selected
    if len(coeffs) == 1:
        plt.title(f'{model_name} Coefficients for "{keyword}" Features in {year}')  
    plt.show()
