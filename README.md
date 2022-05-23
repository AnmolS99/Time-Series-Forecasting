# Time Series Forecasting 📈

This project was part of the Deep Learning (IT3030) course at NTNU spring 2022. The goal of this project was to use time series forecasting to predict transmission system imbalance.

## Installation

To install required packages, use the following command: `pip install -r requirements.txt`

## Datasets

The datasets includes production plans and historical imbalance data for east Norway (NO1 area in the figure below).

![Electricity prices](images/electricityPriceArea.png)

The datasets contain the following features:

- start time: The timestamp of each datum.
- hydro: The planned reservoir hydropower production at the time step.
- micro: The planned small-scale hydropower production at the time step.
- river: The planned run-of-river hydropower production at the time step.
- thermal: The planned thermal power plant production at the time step.
- wind: The planned wind power plant production at the time step.
- total: The total planned production at the time step. Equal to the sum of the listed ”planned production” features.
- sys reg: The planned ”system regulation” at the time step: activation of balancing services to accommodate specific needs (e.g. bottlenecks) in the power system.
- flow: The planned total power flow in or out of the current area of the grid at the time step.
- y: The target variable. The estimated ”open loop” power grid imbalance at the time step. Can be thought of as the imbalance, per area, that would have occurred if balancing services were not activated. It is not possible to observe this value directly, thus it has to be estimated using other measurements of the power grid.
