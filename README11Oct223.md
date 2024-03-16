Name: Loke Git Yan
email: gyloke@yahoo.com.sg
1.  Overview of submitted package: 
    Package comprises files: eda.ipynb, Machine Learning Pipeline (MLP) .py in src folder, requirements.txt, run.sh, and this README.md

2.  Running MLP pipeline: 
    (a) the programme is meant to run by GitHub Actions, it can also be run using run.sh in GitBash environment.  
    (b) pipeline is contained in ShipSail123.py, which incorporates logistic regression/RandomForest/K_neighbors algorithms. 

3.  Modifying the algorithm parameters: 
    a list containing various parameter values is provided for modifying parameter values of each algorithm (however so far I only managed to be successful at this with logistic regression, which regretfully sometimes takes too long to complete run and is thus not included for time being)

4.  Workflow of MLP pipeline: 
    (a) to assign numerical values e.g. 1, 2, 3, 4, 5 to ordinal categorical data; 5 being most important 
    (b) to create a list (num_cols) consists of numerical and ordinal categorical variables from pre_cruise data; to include post_cruise data 'Cruise Distance' in num_cols.
    (c) to create a list (cat_cols) consists of categorical values from cruise_pre data; they are: 'Gender', 'Gate location'
    (d) to create num_pipeline consists of SimpleImputer and MinMaxScaler (or StandardScaler) to receive num_cols data
    (e) to create cat_pipeline consists of SimpleImputer and OneHotEncoder to receive cat_cols data
    (f) to create column transformer col_trans pipeline which runs num_pipeline and cat_pipeline in parallel (n_jobs=-1)
    (g) to create clf_pipeline consists of 2 steps: col_trans and clf model
    (h) to organise data into input data X from num_cols & cat_cols, and target y (Ticket Type), split X and y into train and test dataset
    (i) to run clf_pipeline.fit followed by clf_pipeline.predict for prediction, clf_pipeline.score for accuracy score
    (j) to run command to identify parameters and create grid_params dictionary
    (k) to create GridSearchCV (not run successfully yet, thus not included for time being)
    (l) to run GridSearchCV to get best param, best train score and best test score (not run successfully yet, thus not included for time being)
    
5.  Overview of key findings from the EDA: 
    (a) post_cruise data not to be included as input (X); as being post cruise, they are not available at pre cruise stage yet as part of X to predict y.
    (b) post cruise data 'Cruise Distance' is an exception, as these may have been known pre cruise during Ticket booking. Hence 'Cruise Distance' is included as part of X.
    (c) from heatmap the first 10 highest feature correlation with target in descending order of correlation are: 'Cruise Distance', 'Online Check-in', 'Cabin Comfort','Onboard_Entertainment', 'Cabin Comfort','Baggage handling','Port Check-in Service','Onboard Service','Cleanliness' & 'Age'.
    (d) from heatmap 'Gate location' has little correlation with 'Ticket Type', thus Gate location could have been be dropped from input, there will be little impact on model score
    (e) from crosstab against Ticket Type, 'Luxury' class passengers generally of older age, on longer cruise distance, and they place more emphasis on service, comfort, convenience levels.
    (f) from heatmap 'Gender', 'Gate location', 'Source of Traffic' have little correlation with 'Ticket Type'

6.  Describe how the features in the dataset are processed (summarised in a table)
    (a) Gender - map to: Female: 1 / Male: 2
    (b) Date of Birth - Convert to age in year 2023, na/missing entries are filled by computed mean_age of 39
    (c) Source of Traffic - Map to: company website :3 / email marketing :2 / search engine :1
    (d) Onboard WifiService - Map to: 'Not at all important' :1 / 'A little important': 2 / 'Somewhat important': 3 /'Very important': 4 / 'Extremely important': 5
    (e) Embarkation/Disembarkation time convenient - same as above
    (f) Ease of Online booking	                   - same as above
    (g) Gate location	                           - same as above
    (h) Logging	- (not included due to time constraint, to be included in due course)
    (i) Onboard Dining Service	                   - same as above
    (j) Online Check-in	                           - same as above
    (k) Cabin Comfort	                           - same as above
    (l) Onboard Entertainment	                   - same as above
    (m) Cabin Service	                           - same as above
    (n) Baggage Handling	                       - same as above
    (o) Port Check-in Service	                   - same as above
    (p) Onboard Service	                           - same as above
    (q) Cleanliness	                               - same as above
    (r) Ext_Intcode	                               - dropped, no perceived correlation with target
    (s) Cruise Distance - miles are converted to KM, na/missing values are filled by computed mean_km of 1265 KM

7.  Explanation of choice of models: 
    (a) logistic regression, random forest and K-neighbors are chosen as I am familiar with them and have confidence on their reliability. 
    (b) neural network would have been chosen if not for its persistent errors, complexity in integrating with MLP  

8.  Evaluation of the models:
    (a) logistic regression model test score (accuracy): ~0.65
    (b) random forest model test score (accuracy): ~0.76
    (c) K-neighbors model test score (accuracy): ~0.70
    (d) accuracy is used as model performance metrics for it is a multi class classification problem.
    (e) results from 3 models actually serve to corroborate one another 
    (f) model test scores to be improved by parameter-tuning in due course

9.  Lineplots on these post-cruise categorical variables:
    Passengers' post cruise feedback on 'WiFi', 'Dining', 'Entertainment' are presented as lineplots by the four cruise names: 'blastoise', 'lapras', 'blast', 'lap'. These are for ShipSail management reference in improving servcies. Means of 'blastoise', 'lapras', 'blast', 'lap' are near 0.5; this indicates that only about 50% passengers are satisfied with the facilities. Also it appears that cruise 'lap' scores the lowest among the four cruise names. 