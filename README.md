# Project Summary:

# Predicting Steel Strength: A Regression-based Machine Learning Approach
==============================

_Steel, primarily made from Iron is one of the most important and ubiqutous materials in modern society. In this project, an Emnsemble Regressor, a machine learning model, was developed to predict the strength of a steel sample based on its chemsitry and temperature.
Predicting steel strength from steel chemistry_


## Data

Data was obtained from a now unavailble datadaset on Kaggle 
[Steel dataset](https://www.kaggle.com/datasets/rohannemade/mechanical-properties-of-low-alloy-steels?resource=download)

The dataset has 618 observations. Each of these correspond to a steel sample being pulled at a certain temperature. The machines that performs this strength test, a universal testing machine is shown below. It gathers this data by determining the force (and the resulting stress) required to pull a material up to failure. The sample's strength parameters (Tensile, Yield, Reduction in area, Elongation) are features as well as its chemistry. Yield strength was chsoen to be the target variable so the others were dropped from the dataset. The sample below is the data after cleaning and removing unneccesary features:

A temperature range was chosen between 25˚C and 450˚C. 

![](/reports/figures/utm.png)

![](/reports/figures/data_summary.png)


## EDA

The correlations between the featues and yield strength are shows below:

![](/reports/figures/correlation_map.png)

Four elements, Vanadium (v), Molybdenum (mo), Nickel (ni) and Manganese (mn) play a big positive role in determining strength. Temperature also plays a huge role in negatively influencing strength which is expected as higher temperatures allow for easier movement of dislocations.


## Modelling

The package pycaret was used to perform a preliminary search to find the top models to perform the regression. The models from this analysis would be fed into an ensemble Voting Regressor. The models with the best overall performance were

1. CatBoost Regressor (CAT)
2. Light Gradient Boosting Machine (LGBM)
3. ExtraTrees Regressor (XT)

The CatBoost Regressor, Light Gradient Boosting Machine and Extra Trees Regressor were chosen to be input into a Voting Regressor. In the report, the feature importances of all 3 models is found in the report. Vanadium was common in the CatBoost and XT models while Molybdenum, Nickel and Manganese were common to all three. The LGBM model did put quite a bit of importance on temperature which isn't ideal since most of the samples were pulled at temperates >25˚C.

### Ensemble Model

From these three models, an ensemble Voting Regressor model was built. The model returns a prediction that is the weighted average of the predictions from the three baxe models. The weighted average of the prediction from each regressor is the final prediction from this model. The weights were determined by iteratively assigning a weight (between 0 and 0.9) to each model, similar to a grid search, and retrieving the evaluation metrics from each iteration. Using this method, the optimal weights discovered were 0.3, 0.6 and 0.1 corresponding to CAT, LGBM and XT respectively. This path was chosen because the power in this meta-model comes from its diversity. Any predictions from one model that suffers wheen predicting on one particular section of data can be compensated by the predictions of the other two. It is also more flexible when predicting on different types of data. 

```python
# Weights will be assigned iteratively to each model in a Voting Regressor to discover the most accurate model

# Initialize empty lists for CatBoost weights
weights1 = list()

# Initialize empty list for LGBM weights
weights2 = list()

# Initialize empty list for ExtraTrees weights
weights3 = list()

# Empty list for scoring
mae_loss = list()

# All weights range (0.1,0.9)

# Looping through CatBoost weights
for i in np.arange(0.1,1,0.1):
    
    # Looping through LGBM weights
    for j in np.arange(0.1,1,0.1):
        
        # Looping through ExtraTrees weights
        for k in np.arange(0.1,1,0.1):
            
            # Initializing VotingRegressor with to be determined weights
            vote_reg = VotingRegressor([('cat', cat), ('lgbm', lgbm), ('xt', best_xt)], weights = [i,j,k])
            # Fitting onto training data
            vote_reg.fit(X_train, y_train)
            # Getting predictions
            y_pred = vote_reg.predict(X_test)
            # Getting MAE
            mae_loss = mean_absolute_error(y_pred, y_test)
            
            # Appending scores and weights to respective lists
            loss.append(mae_loss)
            weights1.append(i)
            weights2.append(j)
            weights3.append(k)
```

Shown below is the a diagram of the Voting Regressor structure

![](./reports/figures/ensemble_map.png)


The metrics of each individual model as well as the ensemble Voting Regressor is shown below.

![](./reports/figures/metrics_vote_reg.png)


## Discussion and Conclusion


LGBM contributes 60% to the ensemble model, placing a balanced importance on important elements. It relies on temperature first and foremost but still performs well. Extra Trees has a limited contribution, possibly due to overfitting. Default CatBoost outperforms Voting Regressor, but the latter is still chosen as the final model since it will be more generalizable to new datasets. 

For this business use case, both MAE and RMSE are used to judge the model's performance. Metallurgists need only a rough estimate of steel performance. The Voting Regressor ensemble model had an MAE of ~14.2 MPa, RMSE of ~30 MPa, and an R2 of 0.96. Considering the mean Yield strength of the set is 361 MPa, its performance is excellent for this use case.

An evaluation was done on a subset of the data at a temperature of 27˚C (around room temperature). Firstly, All observations recorded at 27˚C were indexed. Using this index, new X and y datasets were created. These new sets were also removed of any training data. To reiterate, the resulting dataset (25 observations) was comprised exclusively of test and validation data recorded at 27˚C. The results are shown below:


[](/reports/figures/metrics_27.png)






 

When scored on the new test and validation data, the model still performed quite well. It’s MAE was ~27 MPa which is similar to the MAE obtained from training on the data from all temperatures even though the R2 did decrease to 0.91. However, the CV MAE when subsetting for 27˚C was higher than the CV MAE when the data from all temperatures was included (~27 MPa vs. ~14 MPa). 

Due to the reasons specified above, the final model was chosoen to be the ensemble Voting Regressor. 





Project Organization
------------

    ├── LICENSE
    ├── README.md          
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │          
    │
    ├── reports            <- Generated analysis as DOCX, PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
