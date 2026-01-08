## Overall Description 

Based on Daily News for Stock Market Prediction and Dow Jones Industrial Average Prices, 
combine NLP and financial time-series to predict daily stock market direction (up/down).

This project implements a **complete library-agnostic (Sklearn and PyTorch) pipeline** to find best model configurations using **hyperparams random search** among several custom Sklearn / PyTorch models for daily price and associated textual news, and **automatically tracks best metrics and save checkpoints** for future usage.  
Initializing, training and evaluating are done. 

User can add new models and configurations to assess and compare from previous best ones as well. 

Finally, **both** prices and news **models are aggregated to predict market direction on new data**. 

----------

- Custom Cleaning and Tokenization is implemented to match the news dataset
- FinBert pretrained model is used to understand news headlines as a base 
- Sklearn models can be used for prices
- PyTorch models can be used with Tabular / Times series dataset


### Launch

`python main.py -dtype {numerical,textual} -m [model.yaml]`

OR (to run experiments on every model of given lib)

`python main.py -dtype {numerical,textual]}` 

### Directory
- `checkpoints/`: saved model artifacts and configs from previous runs

- `configs/`: experiment settings

- `data/`: raw inputs and processed datasets used for training/testing

- `notebook/`: research and demo notebooks for experiments

- `results/`: aggregated metrics from completed experiments. According to **score**, **datetime** and **model_name**, easily load checkpoint in `checkpoints/`

- `src/`: source code for data loading, preprocessing, models, training and evaluating 

- `models/`: model configurations in YAML


