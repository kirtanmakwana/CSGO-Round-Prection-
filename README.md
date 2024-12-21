---

# CSGO Game Winner Prediction with Heatmap Visualization  

This project uses machine learning to predict the winners of CSGO (Counter-Strike: Global Offensive) games and visualizes the predictions using a heatmap. Several models, including classification algorithms and deep learning, are employed to achieve accurate predictions. All models are implemented and run in a single Jupyter Notebook for ease of use and comparison.  
![CSGO Winner Prediction Winners in feature match](Image.png)  

## Features  

- Utilizes multiple machine learning models:
  - **KNeighborsClassifier**  
  - **RandomizedSearchCV** for hyperparameter optimization  
  - **RandomForestClassifier**  
  - **Sequential (Deep Learning Model)**  
- Provides a detailed visualization of predictions with a heatmap.  
- One-stop implementation in a single Jupyter Notebook.  

## Requirements  

To run this project, you need the following libraries installed:  

- Python 3.x  
- Jupyter Notebook  
- Libraries:  
  ```bash  
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow  
  ```  

## Getting Started  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/csgo-winner-prediction.git  
   ```  

2. Navigate to the project directory:  
   ```bash  
   cd csgo-winner-prediction  
   ```  

3. Open the Jupyter Notebook:  
   ```bash  
   jupyter notebook csgo_prediction.ipynb  
   ```  

4. Follow the steps in the notebook to preprocess data, train models, and visualize results.  

## Project Workflow  

1. **Data Loading and Preprocessing**:  
   - Load the dataset.  
   - Clean and preprocess the data.  

2. **Model Training**:  
   - Train and evaluate various models like KNeighborsClassifier, RandomizedSearchCV, RandomForestClassifier, and Sequential.  
   - Compare the performance of these models.  

3. **Prediction and Visualization**:  
   - Generate predictions using the best-performing model.  
   - Visualize the predictions in a heatmap format.  

## Heatmap Example  

Below is an example heatmap visualization of predicted winners for CSGO games:  

![CSGO Winner Prediction Heatmap](img2.PNG)  

## Conclusion  

This project demonstrates how to use multiple machine learning models to predict the outcome of CSGO games and visualize the results effectively. It provides a comparative analysis of different algorithms and their performance on the same dataset.  

## Acknowledgments  

- **scikit-learn** for machine learning models and evaluation metrics.  
- **TensorFlow** for building and training the deep learning model.  
- **Matplotlib** and **Seaborn** for data visualization.  

Feel free to contribute or report issues!  

---  
