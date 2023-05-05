# Customer_Segmentation_Classification

## Overview

An automobile company wants to enter new markets with their existing products, and they have identified 2627 potential customers in the new markets. In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D). The company plans to use the same strategy for the new markets and have asked for help in predicting the right group of new customers.

This project involves the exploration and analysis of the customer data to develop a machine learning model that can accurately predict the customer segments of the new potential customers. The dataset contains information about the customers such as their age, gender, marital status, education level, occupation, family size, and annual income.

A project like this can be highly beneficial to a company as it provides insights into customer behavior and preferences, allowing for more targeted marketing efforts. By understanding customer segments and tailoring outreach strategies to each group, companies can increase their chances of success in new markets. Additionally, predictive models can assist with identifying potential customers and allocating resources more efficiently. Ultimately, this can lead to improved customer satisfaction, increased revenue, and a stronger competitive edge.

## Getting Started

[Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation)

To get started with this project, you'll need to have the following prerequisites installed on your machine:

Python 3.x
Jupyter Notebook
Required Python packages (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
To install the required packages, you can use pip, the Python package installer, by running the following command:

```pip install numpy pandas matplotlib seaborn scikit-learn```

Once you have all the prerequisites installed, you can clone this repository by running the following command:

```git clone https://github.com/your_username/your_repository.git```

After cloning the repository, navigate to the project directory and open the Jupyter Notebook file named customer_segmentation.ipynb. This file contains all the code for the project, as well as detailed explanations and visualizations of the data analysis and modeling processes.

## Usage

To use this project, simply open the customer_segmentation.ipynb file in Jupyter Notebook and run each code cell in order. The notebook is divided into several sections, each with its own objective and analysis.

The first section covers data loading and preprocessing, where the raw data is read in, explored, and cleaned to prepare it for analysis.

The second section covers exploratory data analysis, where the cleaned data is visualized and summarized to gain insights into the customers' behavior.

The third section covers feature engineering, where new features are created and existing features are transformed to better capture the relationships between the variables.

The fourth section covers modeling, where several machine learning algorithms are trained on the data and evaluated using various metrics to determine the best algorithm for predicting customer segmentation.

The final section concludes the notebook by summarizing the findings and offering recommendations to the automobile company based on the insights gained from the analysis.

You can modify the code and experiment with different machine learning algorithms and parameters to see how they affect the model's performance.

## Data Description

The dataset contains the following columns:

* Customer ID: Unique identifier for each customer
* Age: Age of the customer
* Gender: Gender of the customer (Male/Female)
* Occupation: Occupation of the customer (e.g. Executive, Healthcare, Homemaker, Lawyer, Marketing, None, Professional, Retired, Sales, Student, Technician, Tradesman/Craftsman)
* Marital Status: Marital status of the customer (Married/Single)
* Education Level: Education level of the customer (Graduate/Not Graduate)
* Income: Annual income of the customer
* Family Size: Size of the customer's family
* Var_1: Anonymized variable that can take on different values
* Segmentation: The customer segment to which the customer belongs (A, B, C, D)

## Methodology

The project involves the following steps:

* Data exploration and visualization: Exploring the dataset and visualizing the distribution of different features, correlations between features, and segment distributions.
* Data preprocessing: Cleaning the data, handling missing values, encoding categorical variables, and scaling numerical variables.
* Feature engineering: Creating new features that could potentially improve the model's performance.
* Model selection and training: Selecting the appropriate machine learning algorithm(s) and training the model(s) on the preprocessed data.
* Model evaluation and tuning: Evaluating the performance of the trained model(s) and fine-tuning the hyperparameters to improve their performance.
* Predictions: Using the trained model to predict the customer segments of the new potential customers.

![Age_Dist](https://user-images.githubusercontent.com/80132877/236533795-e33fe538-a220-4c4e-a3af-6c23362bbf26.png)

![gender](https://user-images.githubusercontent.com/80132877/236533810-11831d39-1651-4fd3-be6b-ea3257992fc8.png)

![people_per_segment](https://user-images.githubusercontent.com/80132877/236533826-3d6eb2c5-0fcc-427c-b0f1-fddc76c1e86c.png)

![profession_by_count](https://user-images.githubusercontent.com/80132877/236533842-19407334-70bb-424e-8de9-d7a2ff14cb2a.png)

## Results

After exploring and preprocessing the data, we trained and evaluated four different classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors, and Gradient Boosting. We found that Gradient Boosting provided the best results, with an accuracy of 55% and an F1-score of 54%.

## Conclusion

The results suggest that the trained model could accurately predict the customer segments of new potential customers, albeit with a moderate level of accuracy. Further improvements could be made by exploring additional features or using more advanced machine learning techniques. Overall, the project demonstrates the importance of exploratory data analysis and feature engineering in developing effective machine learning models for customer segmentation.
