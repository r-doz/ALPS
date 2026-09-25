#!/usr/bin/env python3
"""
Results Aggregator
This script aggregates statistics from multiple results.csv files across different models and datasets.
It calculates the mean and standard deviation for each metric and combines them into a single output file.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

def find_results_files(base_dir):
    """Find all results.csv files in the given directory structure"""
    results_files = {}
    
    # Navigate through model folders, skipping seed_* folders
    for model_dir in os.listdir(base_dir):
        # Skip directories that start with "seed_"
        if model_dir.startswith("seed_"):
            continue
            
        model_path = os.path.join(base_dir, model_dir)
        if os.path.isdir(model_path):
            if model_dir not in results_files:
                results_files[model_dir] = {}
                
            # Navigate through dataset folders
            for dataset_dir in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_dir)
                if os.path.isdir(dataset_path):
                    # Look for results.csv file
                    csv_path = os.path.join(dataset_path, "results.csv")
                    if os.path.exists(csv_path):
                        results_files[model_dir][dataset_dir] = csv_path
    
    return results_files

def aggregate_results(results_files):
    """
    Aggregate statistics from results files
    
    Returns:
    - aggregated_stats: Dictionary with mean and std dev statistics
    - all_metrics: List of all metrics found in results files
    """
    aggregated_stats = {}
    all_metrics = set()
    
    # Process each model and dataset
    for model, datasets in results_files.items():
        if model not in aggregated_stats:
            aggregated_stats[model] = {}
            
        for dataset, file_path in datasets.items():
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                if len(df) == 0:
                    print(f"Warning: Empty CSV file for {model}/{dataset}")
                    continue
                
                # Identify numeric columns for statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                all_metrics.update(numeric_cols)
                
                # Calculate statistics for each numeric column
                stats = {}
                for col in numeric_cols:
                    # Drop NaN values for this column
                    valid_values = df[col].dropna()
                    
                    if len(valid_values) > 0:
                        stats[col] = {
                            'mean': valid_values.mean(),
                            'std': valid_values.std(),
                            'count': len(valid_values)
                        }
                
                # Store the statistics
                aggregated_stats[model][dataset] = stats
                print(f"Processed {model}/{dataset}: {len(numeric_cols)} metrics, {len(df)} rows")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return aggregated_stats, list(all_metrics)

def create_aggregated_dataframe(aggregated_stats, all_metrics):
    """Create a DataFrame with the aggregated statistics"""
    rows = []
    
    for model, datasets in aggregated_stats.items():
        for dataset, metrics in datasets.items():
            # Create a row for mean values
            mean_row = {
                'model': model,
                'dataset': dataset,
                'stat_type': 'mean'
            }
            
            # Create a row for std deviation values
            std_row = {
                'model': model,
                'dataset': dataset,
                'stat_type': 'std'
            }
            
            # Create a row for count values
            count_row = {
                'model': model,
                'dataset': dataset,
                'stat_type': 'count'
            }
            
            # Add metrics to rows
            for metric in all_metrics:
                if metric in metrics:
                    mean_row[metric] = metrics[metric]['mean']
                    std_row[metric] = metrics[metric]['std']
                    count_row[metric] = metrics[metric]['count']
            
            rows.append(mean_row)
            rows.append(std_row)
            rows.append(count_row)
    
    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by model and dataset
    df = df.sort_values(['model', 'dataset', 'stat_type'])
    
    return df

def create_pivoted_dataframe(aggregated_stats, all_metrics):
    """Create a pivoted DataFrame where each model-dataset has a single row"""
    rows = []
    
    for model, datasets in aggregated_stats.items():
        for dataset, metrics in datasets.items():
            # Create a row for this model-dataset
            row = {
                'model': model,
                'dataset': dataset
            }
            
            # Add metrics to row
            for metric in all_metrics:
                if metric in metrics:
                    row[f"{metric}_mean"] = metrics[metric]['mean']
                    row[f"{metric}_std"] = metrics[metric]['std']
                    row[f"{metric}_count"] = metrics[metric]['count']
            
            rows.append(row)
    
    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by model and dataset
    df = df.sort_values(['model', 'dataset'])
    
    return df

def format_excel_file(df, file_path, pivot_style=False):
    """Create a formatted Excel file with the aggregated statistics"""
    # Save to Excel
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Aggregated Stats')
    
    # Access the workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Aggregated Stats']
    
    # Set column widths - only for A-Z columns to avoid errors
    max_safe_column = min(len(df.columns), 26)  # Only columns A-Z
    for i in range(max_safe_column):
        col_letter = chr(65 + i)  # Convert to A, B, C, etc.
        col_width = max(len(str(df.columns[i])), 15)
        worksheet.column_dimensions[col_letter].width = col_width
    
    # Format headers
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))
    
    for col in range(1, len(df.columns) + 1):
        cell = worksheet.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal='center')
    
    # Apply formatting based on the content
    if pivot_style:
        # For pivoted style: highlight metric types (mean/std/count)
        for row in range(2, len(df) + 2):
            for col in range(1, len(df.columns) + 1):
                cell = worksheet.cell(row=row, column=col)
                cell.border = thin_border
                
                # Color coding for different metric types - only apply to column headers we can access
                if col <= max_safe_column:
                    header_value = str(worksheet.cell(row=1, column=col).value)
                    if "_mean" in header_value:
                        cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
                    elif "_std" in header_value:
                        cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    else:
        # For regular style: highlight different stat types (mean/std/count)
        row_fills = {
            'mean': PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid"),
            'std': PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid"),
            'count': PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
        }
        
        for row in range(2, len(df) + 2):
            if 'stat_type' in df.columns:
                stat_type = str(df.iloc[row-2]['stat_type'])
                
                for col in range(1, len(df.columns) + 1):
                    cell = worksheet.cell(row=row, column=col)
                    cell.border = thin_border
                    
                    if stat_type in row_fills and col > 3:  # Skip the first three columns (model, dataset, stat_type)
                        cell.fill = row_fills[stat_type]
    
    # Save the workbook
    writer.close()

def main(base_dir, output_dir):
    """Main function to aggregate results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for results files in {base_dir}...")
    results_files = find_results_files(base_dir)
    
    # Count total files found
    total_files = sum(len(datasets) for datasets in results_files.values())
    print(f"Found {total_files} results.csv files across {len(results_files)} models")
    
    if total_files == 0:
        print("No results files found. Exiting.")
        return
    
    # Aggregate statistics
    print("Aggregating statistics...")
    aggregated_stats, all_metrics = aggregate_results(results_files)
    
    # Create regular style DataFrame (with separate rows for mean/std)
    df_regular = create_aggregated_dataframe(aggregated_stats, all_metrics)
    
    # Create pivoted style DataFrame (with mean/std in columns)
    df_pivoted = create_pivoted_dataframe(aggregated_stats, all_metrics)
    
    # Save to CSV
    regular_csv_path = os.path.join(output_dir, "aggregated_stats.csv")
    pivoted_csv_path = os.path.join(output_dir, "aggregated_stats_pivoted.csv")
    
    df_regular.to_csv(regular_csv_path, index=False)
    df_pivoted.to_csv(pivoted_csv_path, index=False)
    
    print(f"Saved regular format to {regular_csv_path}")
    print(f"Saved pivoted format to {pivoted_csv_path}")
    
    # Save to Excel with formatting
    regular_excel_path = os.path.join(output_dir, "aggregated_stats.xlsx")
    pivoted_excel_path = os.path.join(output_dir, "aggregated_stats_pivoted.xlsx")
    
    format_excel_file(df_regular, regular_excel_path, pivot_style=False)
    format_excel_file(df_pivoted, pivoted_excel_path, pivot_style=True)
    
    print(f"Saved formatted Excel to {regular_excel_path}")
    print(f"Saved formatted Excel (pivoted) to {pivoted_excel_path}")
    
    # Generate summary of available metrics
    metrics_summary = {
        "total_models": len(results_files),
        "total_datasets": sum(len(datasets) for model, datasets in results_files.items()),
        "total_metrics": len(all_metrics),
        "metrics": all_metrics,
        "model_dataset_coverage": {}
    }
    
    for model, datasets in results_files.items():
        metrics_summary["model_dataset_coverage"][model] = list(datasets.keys())
    
    # Save summary to JSON
    import json
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"Saved metrics summary to {summary_path}")
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate statistics from results.csv files")
    parser.add_argument("--base_dir", "-b", required=True, help="Base directory containing model/dataset folders")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for aggregated statistics")
    
    args = parser.parse_args()
    
    main(args.base_dir, args.output_dir)