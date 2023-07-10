Predicting Steel Strength: A Regression-based Machine Learning Approach
==============================

_Steel, primarily made from Iron is one of the most important and ubiqutous materials in modern society. In this project, an Emnsemble Regressor, a machine learning model, was developed to predict the strength of a steel sample based on its chemsitry and temperature.
Predicting steel strength from steel chemistry_


## Data

Data was obtained from a now unavailble datadaset on Kaggle 
[Steel dataset](https://www.kaggle.com/datasets/rohannemade/mechanical-properties-of-low-alloy-steels?resource=download)

The dataset has 618 observations. Each of these correspond to a steel sample being pulled at a certain temperature. The sample's strength parameters (Tensile, Yield, Reduction in area, Elongation) are features as well as its chemistry. Yield strength was chsoen to be the target variable so the others were dropped from the dataset. The sample below is the data after cleaning and removing unneccesary features:

A temperature range was chosen between 25˚C and 450˚C. 

![](./reports/images/data_summary.png)


## EDA

The correlations between the featues and yield strength are shows below:

![](./reports/images/correlation_map.png)

Four elements, Vanadium (v), Molybdenum (mo), Nickel (ni) and Manganese (mn) play a big positive role in determining strength. Temperature also plays a huge role in negatively influencing strength which is expected as higher temperatures allow for easier movement of dislocations.


## Modelling

The package pycaret was used to perform a preliminary search to find the top models to perform the regression. The models from this analysis would be fed into an ensemble Voting Regressor. The models with the best overall performance were

1. CatBoost Regressor (CAT)
2. Light Gradient Boosting Machine (LGBM)
3. ExtraTrees Regressor (XT)

The CatBoost Regressor, Light Gradient Boosting Machine and Extra Trees Regressor were chosen to be input into a Voting Regressor. In the report, the feature importances of all 3 models is found in the report. Vanadium was common in the CatBoost and XT models while Molybdenum, Nickel and Manganese were common to all three. The LGBM model did put quite a bit of importance on temperature which isn't ideal since most of the samples were pulled at temperates >25˚C.

### Ensemble Model

From these three models, an ensemble Voting Regressor model was built. The weighted average of the prediction from each regressor is the final prediction from this model.
The weights were determined by iteratively assigning a weight (between 0 and 0.9) to each model, similar to a grid search, and retrieving the evaluation metrics from each iteration. Using this method, the optimal weights were 0.7, 0.1 and 0.2 corresponding to CAT, LGBM and XT respectively. 

```python
# Initialize empty lists for CatBoost weights
# Initialize empty lists for CatBoost weights
weights1 = []

# Initialize empty list for LGBM weights
weights2 = []

# Initialize empty list for ExtraTrees weights
weights3 = []

# Empty list for scoring
scores = []

# All weights range (0.1,0.9)

# Looping through CatBoost weights
for i in np.arange(0.1,1,0.1):
    
    # Looping through LGBM weights
    for j in np.arange(0.1,1,0.1):
        
        # Looping through ExtraTrees weights
        for k in np.arange(0.1,1,0.1):
            
            # Initializing VotingRegressor with to be determined weights
            vote_reg = VotingRegressor([('cat', cat), ('lgbm', best_lgbm), ('xt', best_xt)], weights = [i,j,k])
            # Fitting onto training data
            vote_reg.fit(X_train, y_train)
            # Getting prediction
            y_pred = vote_reg.predict(X_test)
            # Getting r2 score
            score = r2_score(y_pred, y_test)
            
            # Appending scores and weights to respective lists
            scores.append(score)
            weights1.append(i)
            weights2.append(j)
            weights3.append(k)
```

Below the metrics of the individual model as well as the ensemble model.


Shown below is the method the Voting Regressor used for predictions

![](./reports/images/ensemble_map.png)



![](./reports/images/metrics_vote_reg.png)

The model The low weght 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
