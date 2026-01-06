"""
EDA Analyser Module.

This module contains the EDAAnalyser class for performing exploratory data analysis
and generating visualizations for the California housing dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import FIGURE_DPI, FIGURE_SIZE, FIGURES_DIR, PLOT_STYLE, TARGET_COLUMN


class EDAAnalyser:
    """
    Performs exploratory data analysis and generates visualizations.

    This class provides methods for creating 10 different types of visualizations:
    1. Histogram
    2. Box plot
    3. Scatter plot
    4. Correlation heatmap
    5. Pair plot
    6. Bar chart
    7. Violin plot
    8. Line chart
    9. Density plot (KDE)
    10. Geographic scatter

    Attributes:
        data (pd.DataFrame): Data to analyze
        figures_dir (Path): Directory to save figures
    """

    def __init__(self, data: pd.DataFrame, figures_dir: Path = None):
        """
        Initialize EDA analyser.

        Args:
            data (pd.DataFrame): Data for analysis
            figures_dir (Path, optional): Directory for saving figures.
                If None, uses FIGURES_DIR from config.

        Raises:
            ValueError: If data is None or empty
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")

        self.data = data
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR

        # Ensure directory exists
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        try:
            plt.style.use(PLOT_STYLE)
        except:
            # Fallback if style not available
            sns.set_style("darkgrid")

        print(f"EDAAnalyser initialized with {len(data)} rows and {len(data.columns)} columns")

    def plot_histogram(self, column: str = 'median_house_value',
                      bins: int = 50, save: bool = True) -> None:
        """
        Plot histogram for specified column (Visualization #1).

        Shows the distribution of values for a numeric column.

        Args:
            column (str): Column name to plot (default: 'median_house_value')
            bins (int): Number of bins (default: 50)
            save (bool): Whether to save figure (default: True)

        Raises:
            KeyError: If column doesn't exist
        """
        plt.figure(figsize=FIGURE_SIZE)

        plt.hist(self.data[column], bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_val = self.data[column].mean()
        median_val = self.data[column].median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        plt.legend()

        if save:
            filename = self.figures_dir / f'01_histogram_{column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_boxplot(self, x_column: str = 'income_category',
                    y_column: str = 'median_house_value',
                    save: bool = True) -> None:
        """
        Plot boxplot showing distribution across categories (Visualization #2).

        Shows how a numeric variable varies across different categories.

        Args:
            x_column (str): Categorical column (x-axis)
            y_column (str): Numeric column (y-axis)
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        plt.figure(figsize=FIGURE_SIZE)

        sns.boxplot(data=self.data, x=x_column, y=y_column, palette='Set2')
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(y_column.replace('_', ' ').title())
        plt.title(f'{y_column.replace("_", " ").title()} by {x_column.replace("_", " ").title()}')
        plt.xticks(rotation=45)

        if save:
            filename = self.figures_dir / f'02_boxplot_{y_column}_by_{x_column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_scatter(self, x_column: str = 'longitude',
                    y_column: str = 'latitude',
                    hue: str = None,
                    save: bool = True) -> None:
        """
        Plot scatter plot with optional color coding (Visualization #3).

        Shows relationship between two numeric variables.

        Args:
            x_column (str): X-axis column
            y_column (str): Y-axis column
            hue (str, optional): Column for color coding
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        plt.figure(figsize=FIGURE_SIZE)

        if hue:
            sns.scatterplot(data=self.data, x=x_column, y=y_column, hue=hue,
                          palette='viridis', alpha=0.6)
        else:
            plt.scatter(self.data[x_column], self.data[y_column],
                       alpha=0.6, color='steelblue')

        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(y_column.replace('_', ' ').title())
        plt.title(f'{y_column.replace("_", " ").title()} vs {x_column.replace("_", " ").title()}')

        if save:
            hue_suffix = f'_by_{hue}' if hue else ''
            filename = self.figures_dir / f'03_scatter_{x_column}_vs_{y_column}{hue_suffix}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_correlation_heatmap(self, save: bool = True) -> None:
        """
        Plot correlation heatmap for all numeric features (Visualization #4).

        Shows correlations between all numeric variables.

        Args:
            save (bool): Whether to save figure
        """
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])

        plt.figure(figsize=(14, 10))

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

        plt.title('Correlation Heatmap of Numeric Features')
        plt.tight_layout()

        if save:
            filename = self.figures_dir / '04_correlation_heatmap.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_pairplot(self, columns: List[str] = None,
                     hue: str = 'income_category',
                     save: bool = True) -> None:
        """
        Plot pairplot for specified columns (Visualization #5).

        Shows pairwise relationships between multiple variables.

        Args:
            columns (list, optional): Columns to include.
                If None, uses key features.
            hue (str, optional): Column for color coding
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        if columns is None:
            # Select key features for pairplot to keep it manageable
            columns = [
                'median_income', 'median_house_value',
                'housing_median_age', 'rooms_per_household'
            ]

        # Add hue column if not already in columns
        if hue and hue not in columns:
            plot_data = self.data[columns + [hue]]
        else:
            plot_data = self.data[columns]

        # Create pairplot
        pairplot = sns.pairplot(plot_data, hue=hue, palette='Set2',
                               diag_kind='kde', plot_kws={'alpha': 0.6})

        pairplot.fig.suptitle('Pairwise Relationships of Key Features', y=1.02)

        if save:
            filename = self.figures_dir / '05_pairplot.png'
            pairplot.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_bar_chart(self, x_column: str = 'income_category',
                      y_column: str = 'median_house_value',
                      aggregation: str = 'mean',
                      save: bool = True) -> None:
        """
        Plot bar chart with aggregated values (Visualization #6).

        Shows aggregated statistics across categories.

        Args:
            x_column (str): Categorical column
            y_column (str): Numeric column to aggregate
            aggregation (str): 'mean', 'sum', 'count', 'median'
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
            ValueError: If aggregation method is invalid
        """
        plt.figure(figsize=FIGURE_SIZE)

        # Perform aggregation
        if aggregation == 'mean':
            agg_data = self.data.groupby(x_column)[y_column].mean()
        elif aggregation == 'sum':
            agg_data = self.data.groupby(x_column)[y_column].sum()
        elif aggregation == 'count':
            agg_data = self.data.groupby(x_column)[y_column].count()
        elif aggregation == 'median':
            agg_data = self.data.groupby(x_column)[y_column].median()
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        # Create bar chart
        agg_data.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(f'{aggregation.title()} {y_column.replace("_", " ").title()}')
        plt.title(f'{aggregation.title()} {y_column.replace("_", " ").title()} by {x_column.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        if save:
            filename = self.figures_dir / f'06_bar_{aggregation}_{y_column}_by_{x_column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_violin(self, x_column: str = 'age_category',
                   y_column: str = 'median_income',
                   save: bool = True) -> None:
        """
        Plot violin plot showing distribution density (Visualization #7).

        Combines aspects of box plots and kernel density plots.

        Args:
            x_column (str): Categorical column
            y_column (str): Numeric column
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        plt.figure(figsize=FIGURE_SIZE)

        sns.violinplot(data=self.data, x=x_column, y=y_column, palette='muted')
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(y_column.replace('_', ' ').title())
        plt.title(f'Distribution of {y_column.replace("_", " ").title()} by {x_column.replace("_", " ").title()}')
        plt.xticks(rotation=45)

        if save:
            filename = self.figures_dir / f'07_violin_{y_column}_by_{x_column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_line_chart(self, x_column: str = 'housing_median_age',
                       y_column: str = 'median_house_value',
                       aggregation: str = 'mean',
                       save: bool = True) -> None:
        """
        Plot line chart showing trends (Visualization #8).

        Shows how a variable changes across another variable.

        Args:
            x_column (str): X-axis column
            y_column (str): Y-axis column
            aggregation (str): Aggregation method for grouping
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        plt.figure(figsize=FIGURE_SIZE)

        # Group and aggregate
        if aggregation == 'mean':
            line_data = self.data.groupby(x_column)[y_column].mean()
        elif aggregation == 'median':
            line_data = self.data.groupby(x_column)[y_column].median()
        else:
            line_data = self.data.groupby(x_column)[y_column].mean()

        # Plot line chart
        plt.plot(line_data.index, line_data.values, marker='o', linewidth=2,
                markersize=6, color='steelblue')
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(f'{aggregation.title()} {y_column.replace("_", " ").title()}')
        plt.title(f'Trend of {y_column.replace("_", " ").title()} by {x_column.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)

        if save:
            filename = self.figures_dir / f'08_line_{y_column}_by_{x_column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_density(self, columns: List[str] = None, save: bool = True) -> None:
        """
        Plot kernel density estimation for multiple columns (Visualization #9).

        Shows probability density functions for comparison.

        Args:
            columns (list, optional): Columns to plot densities for.
                If None, uses key numeric features.
            save (bool): Whether to save figure

        Raises:
            KeyError: If columns don't exist
        """
        if columns is None:
            columns = ['median_income', 'rooms_per_household', 'population_per_household']

        plt.figure(figsize=FIGURE_SIZE)

        for column in columns:
            if column in self.data.columns:
                self.data[column].plot(kind='kde', label=column.replace('_', ' ').title(),
                                      linewidth=2)

        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Density Plots of Selected Features')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            filename = self.figures_dir / '09_density_multiple_features.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_geographic_scatter(self, color_column: str = 'median_house_value',
                               save: bool = True) -> None:
        """
        Plot geographic scatter with color representing values (Visualization #10).

        Creates a map-like visualization of California housing data.

        Args:
            color_column (str): Column to use for color scale
            save (bool): Whether to save figure

        Raises:
            KeyError: If required columns don't exist
        """
        plt.figure(figsize=(12, 10))

        # Create scatter plot with geographic coordinates
        scatter = plt.scatter(
            self.data['longitude'],
            self.data['latitude'],
            c=self.data[color_column],
            cmap='YlOrRd',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'California Housing: {color_column.replace("_", " ").title()}')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_column.replace('_', ' ').title())

        # Set aspect ratio to match California's geography
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, alpha=0.3)

        if save:
            filename = self.figures_dir / f'10_geographic_scatter_{color_column}.png'
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def generate_all_plots(self) -> Dict[str, str]:
        """
        Generate all 10 visualization types at once.

        This is a convenience method that creates all visualizations
        and saves them to the figures directory.

        Returns:
            dict: Mapping of plot names to file paths
        """
        print("Generating all visualizations...")
        print("="*50)

        plots = {}

        try:
            # 1. Histogram
            print("\n[1/10] Generating histogram...")
            self.plot_histogram()
            plots['histogram'] = str(self.figures_dir / '01_histogram_median_house_value.png')

            # 2. Boxplot
            print("[2/10] Generating boxplot...")
            self.plot_boxplot()
            plots['boxplot'] = str(self.figures_dir / '02_boxplot_median_house_value_by_income_category.png')

            # 3. Scatter plot
            print("[3/10] Generating scatter plot...")
            self.plot_scatter()
            plots['scatter'] = str(self.figures_dir / '03_scatter_longitude_vs_latitude.png')

            # 4. Correlation heatmap
            print("[4/10] Generating correlation heatmap...")
            self.plot_correlation_heatmap()
            plots['heatmap'] = str(self.figures_dir / '04_correlation_heatmap.png')

            # 5. Pairplot
            print("[5/10] Generating pairplot...")
            self.plot_pairplot()
            plots['pairplot'] = str(self.figures_dir / '05_pairplot.png')

            # 6. Bar chart
            print("[6/10] Generating bar chart...")
            self.plot_bar_chart()
            plots['bar'] = str(self.figures_dir / '06_bar_mean_median_house_value_by_income_category.png')

            # 7. Violin plot
            print("[7/10] Generating violin plot...")
            self.plot_violin()
            plots['violin'] = str(self.figures_dir / '07_violin_median_income_by_age_category.png')

            # 8. Line chart
            print("[8/10] Generating line chart...")
            self.plot_line_chart()
            plots['line'] = str(self.figures_dir / '08_line_median_house_value_by_housing_median_age.png')

            # 9. Density plot
            print("[9/10] Generating density plot...")
            self.plot_density()
            plots['density'] = str(self.figures_dir / '09_density_multiple_features.png')

            # 10. Geographic scatter
            print("[10/10] Generating geographic scatter...")
            self.plot_geographic_scatter()
            plots['geographic'] = str(self.figures_dir / '10_geographic_scatter_median_house_value.png')

            print("\n" + "="*50)
            print("All visualizations generated successfully!")
            print(f"Saved to: {self.figures_dir}")
            print("="*50)

        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            raise

        return plots

    def get_correlation_analysis(self, target: str = None) -> pd.Series:
        """
        Get correlation coefficients with target variable.

        Args:
            target (str, optional): Target variable name.
                If None, uses TARGET_COLUMN from config.

        Returns:
            pd.Series: Correlations sorted by absolute value (descending)
        """
        if target is None:
            target = TARGET_COLUMN

        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])

        # Calculate correlations
        correlations = numeric_data.corr()[target].sort_values(ascending=False)

        print(f"\nCorrelations with {target}:")
        print("="*50)
        print(correlations)
        print("="*50)

        return correlations

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all numeric columns.

        Returns:
            pd.DataFrame: Summary statistics (count, mean, std, min, quartiles, max)
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        summary = numeric_data.describe()

        print("\nSummary Statistics:")
        print("="*50)
        print(summary)
        print("="*50)

        return summary
