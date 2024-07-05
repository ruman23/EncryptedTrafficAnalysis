# Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol

## Project Overview

This repository contains scripts and a limited dataset used for the paper titled **"Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol"**.

## Folder Structure

- **Data/**: This folder contains the extracted RRB dataset.
- **RandomForestClassifier.py**: This script performs classification using the Random Forest algorithm.
- **ExtraTreesClassifier.py**: This script performs classification using the Extra Trees algorithm.
- **ExtractFeaturesForMLModels.py**: This script processes the extracted RRB datasets and calculates the statistical features for machine learning models.
- **data.csv**: This is the output file generated by `ExtractFeaturesForMLModels.py`, containing the processed data with extracted features.



## File Descriptions

### 1. ExtractFeaturesForMLModels.py

**Purpose**: This script processes the raw datasets and extracts statistical features such as mean, max, standard deviation, slope, and quartiles for both `UL` and `DL`. The extracted features are saved in `data.csv`.

**How to Run**:
```sh
python ExtractFeaturesForMLModels.py
```

# Machine Learning Classification Scripts

This repository contains scripts for classification using the Random Forest and Extra Trees algorithms. The scripts use extracted features from `data.csv` to perform the classification and evaluate the classifier's performance.

## Scripts

1. **RandomForestClassifier.py**
   - **Purpose:** This script performs classification using the Random Forest algorithm. It evaluates the classifier's performance and prints the accuracy and classification report.
   - **How to Run:**
     ```sh
     python RandomForestClassifier.py
     ```

2. **ExtraTreesClassifier.py**
   - **Purpose:** This script performs classification using the Extra Trees algorithm. It also does a similar analysis of RandomForestClassifier.
   - **How to Run:**
      ```sh
     python ExtraTreesClassifier.py
      ```

## Required Libraries

To run these scripts, you need to install the following Python libraries:
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib

You can install these libraries using pip: 
 ```sh
pip install pandas numpy scikit-learn scipy matplotlib
```

If you face any issues or have any queries, please refer to the paper associated with this work or feel free to contact any of the authors.

## License

This work is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. 

You are free to:

- **Share** — copy and redistribute the material in any medium or format for any purpose, even commercially.
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

- **Attribution** — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more details, please refer to the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/deed.en).


### Disclaimer
The authors of this work expressly prohibit the misuse of this material or the ideas contained within. This work is intended for academic, research, and commercial purposes as allowed under the license. Any misuse of this material or ideas is strictly prohibited. The authors are not responsible for any misuse of the provided materials or ideas.

## Issues and Queries

If you face any issues or have any queries, please refer to the paper associated with this work or feel free to contact any of the authors:

- **Md Ruman Islam**
  - University of Nebraska Omaha
  - Email: mdrumanislam@unomaha.edu

- **Raja Hasnain Anwar**
  - University of Massachusetts Amherst
  - Email: ranwar@umass.edu

- **Spyridon Mastorakis**
  - University of Notre Dame
  - Email: mastorakis@nd.edu

- **Muhammad Taqi Raza**
  - University of Massachusetts Amherst
  - Email: taqi@umass.edu
