# The Gaze of Empathy: Predicting Empathy Scores with Interpretable Machine Learning Models Based on Eye-Movement Activities

As for CE888 assignment2, this project aims to predict empathy scores based on eye-movement features using machine learning techniques. The model is trained on a EyeT4Empathy dataset.

## Dataset
The dataset contains:
- [EyeT data](https://figshare.com/articles/dataset/Eye_Tracker_Data/19729636/2): make sure to place these data in the `data/EyeT` directory.
- [Labels](https://figshare.com/articles/dataset/Questionnaires/19657323/2): make sure to place these data in the `data/label` directory.

## Dependencies
To run this project, you'll need the following Python libraries:
- pandas
- numpy
- tqdm
- scikit-learn
- scipy
- matplotlib
- seaborn

You can install the required libraries using pip:
```
pip install -r requirements.txt
```

## Usage
Before training models, you need to first prepare the dataset by executing the following command in the terminal:

```
python prepare_data.py
```

Then the training and test dataset are placed in `data/train/` and `data/test/`. Also dataframes are provided in `data/data_df.csv` and `data/test_data_df.csv`, respectively.

And to train and evaluate the machine learning models, you can run the main script by executing the following command in the terminal

```
python main.py
```

## Model evaluation
After running the main.py script, you will see the evaluation metrics displayed in the terminal. This will help you assess the performance of the model and make improvements as needed.


## Visualization
To show the visualization for feature selection and evaluation plots, you can refer to the `visualization.ipynb` file.

