Ghaouthi Brahim 22102881  

Lema Xavier Santiago 22108098  

sas-project

https://mygit.th-deg.de/sl10098/sas-project

# Project Description

Given datasets (obtained in Kaggle) of different solar energy plants in India, we aim to predict how much energy could be generated given certain input.
Rows are measurements taken over a period of 30 days every 15 minutes.

# Installation

To run the app the required libraries should be installed first:

```sh
pip install -r requirements.txt
```

If the dependencies are fulfilled, just execute the [main.py](main.py) file to run the PyQt app.

# Basic Usage

The usage is very simple just set up a irradiation level and hit the predict button to see how much power would be produced by the plant.

> Note about the lack of further parameters: we only selected the irradiation parameter because it is directly related to the amount of power produced. Adding more attributes to the model resulted in worse predictions. We are aware that the amount of parameters is less than what is required, but we wanted to work with an organic set of data, and the selection of the dataset was not optimal. We perpetuated an error we had at the beginning. After preparing the data and running some predictions, we noticed the direct correlation between irradiation and energy produced by solar panels. Maybe a fitter dataset for the purpose of this project would have been a dataset with no radiation values but other attributes such as weather, amount of panels, position of the sun (and so on), but given the deadlines and the work already done we proceeded with the development of the app.

# Implementation of the Requests

The app can help you predict the amount of direct current power which would be generated by the solar plant

# Work done

Data was separated in two different datasets, and there was some irrelevant data so we had to do some data cleaning and merge the datasets.

At the same time, for each plant dataset we had two different tables: one for sensors and the other for generators.

There were other challenges for example date-time strings didn't have all the same format, so we have to do some data cleaning before merging into a unique dataframe.

Further details on the data preparation on the notebook [data_preparation.ipynb](data_preparation.ipynb)

Then we created some visualizations with seaborn and matplotlib and implemented a linear regression algorithm to determine the resulting generated power value for a certain user input.

A user interface was created to handle user inputs and display them in an graph.
