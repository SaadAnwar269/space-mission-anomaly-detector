This is a prototype implementation for the Space Mission Risk Analysis Project in the REYES Program.

The datasets used are provided in the respective folder
The file is a simple python script and does not require any additional dependencies, can be run directly from space.py 
The Outputs from the scripts I've ran are also placed in their respective folder

Methodology:

1. Initially the dataset I chose was the NASA CMAPSS however due to it being officially unavailable, I have used a dataset from Kaggle which has dummy data for space missions.
2. The specialized anomaly detection pipeline starts with preprocessing and data cleaning in which I have first encoded the relevant categorical attributes and standardized the numerical features of the data.
3. After Preparation of the data, I have used Isolation Forest to detect anomalies with the contamination initially set to 0.05 or 5%. The test was run with different values of contamination to detect anomlies on different scales.
4. After Detecting Anomalous data, the outliers were visualized via a comparison graph.
5. Alternatively, instead of relying on just the Isolation forest, rule based anomalies were also detected which were manually adjusted to get desired out.

Dataset Link:
https://www.kaggle.com/datasets/sameerk2004/space-missions-dataset?resource=download


