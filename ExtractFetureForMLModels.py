import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def remove_noise(data):
    # Remove zeros
    data_no_zeros = data[data != 0]
    
    # Compute the interquartile range (IQR) and use it to define outliers
    Q1 = np.percentile(data_no_zeros, 25)
    Q3 = np.percentile(data_no_zeros, 75)
    IQR = Q3 - Q1
    
    # Define bounds for the non-outlier data
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 2.0 * IQR
    
    # Adjust values above the upper bound to the upper bound or another specified value
    # For example, cap values to the upper bound, or you could use a higher percentile (e.g., 95th percentile)
    cap_value = np.percentile(data_no_zeros, 95)  # Adjusting to the 95th percentile as an example
    data_adjusted = np.where(data_no_zeros > upper_bound, cap_value, data_no_zeros)
    
    return data_adjusted

def calculate_statistics(data, title):
    """Calculate required statistics for a given dataset with noise reduction."""
    if data.empty:
        print("Error: Data is empty")
        return [np.nan]*6  # Return NaNs for all statistics if data is empty
    
    # Flatten the DataFrame to a 1D array and remove noise
    flattened_data = data.values.flatten()
    filtered_data = remove_noise(flattened_data)

    
    if len(filtered_data) == 0:
        print("Error: All data filtered out as noise")
        return [np.nan]*6
    
    mean_val = np.mean(filtered_data)
    max_val = np.max(filtered_data)
    std_val = np.std(filtered_data)
    
    # Perform linear regression on the filtered array if it contains more than one value
    if len(filtered_data) > 1:
        slope, _, _, _, _ = linregress(np.arange(len(filtered_data)), filtered_data)
    else:
        slope = np.nan
    
    first_q = np.percentile(filtered_data, 25)
    third_q = np.percentile(filtered_data, 75)
    
    return mean_val, max_val, std_val, slope, first_q, third_q



def process_files(dataFolder, output_path):
    aggregated_data = []

    for main_category in sorted(os.listdir(dataFolder)):
        main_category_path = os.path.join(dataFolder, main_category)
        if os.path.isdir(main_category_path) and main_category != '.DS_Store':
            for sub_category in sorted(os.listdir(main_category_path)):
                sub_category_path = os.path.join(main_category_path, sub_category)
                if os.path.isdir(sub_category_path) and sub_category != '.DS_Store':
                    iterations = set(f.split('.')[0] for f in os.listdir(os.path.join(sub_category_path, 'UL')) if f.endswith('.csv'))
                    iterations.update(f.split('.')[0] for f in os.listdir(os.path.join(sub_category_path, 'DL')) if f.endswith('.csv'))

                    for iteration in sorted(iterations, key=int):
                        row = {
                            'iteration': iteration,
                            'maincategory': main_category,
                            'subcategory': sub_category
                        }
                        for feature_type in ['UL', 'DL']:
                            dir_path = os.path.join(sub_category_path, feature_type)
                            file_path = os.path.join(dir_path, f"{iteration}.csv")
                            if os.path.exists(file_path):
                                try:
                                    data = pd.read_csv(file_path, header=None)
                                    stats = calculate_statistics(data,file_path)
                                    for stat_name, stat_value in zip(['mean', 'max', 'std', 'slope', 'firstquartile', 'thirdquartile'], stats):
                                        row[f'{feature_type.lower()}{stat_name}'] = stat_value
                                except Exception as e:
                                    print(f"Error reading {file_path}: {e}")
                                    for stat in ['mean', 'max', 'std', 'slope', 'firstquartile', 'thirdquartile']:
                                        row[f'{feature_type.lower()}{stat}'] = np.nan
                            else:
                                for stat in ['mean', 'max', 'std', 'slope', 'firstquartile', 'thirdquartile']:
                                    row[f'{feature_type.lower()}{stat}'] = np.nan
                        
                        aggregated_data.append(row)

    # Convert the aggregated data to a DataFrame and save to CSV
    final_df = pd.DataFrame(aggregated_data)
    if not final_df.empty:
        final_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    else:
        print("No data was processed. Please check the dataset and script.")

dataFolder = 'Data'  # Update with your actual data folder path
output_path = 'data.csv'  # Update with your actual output path
process_files(dataFolder, output_path)
