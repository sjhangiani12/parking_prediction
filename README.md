# Parking Capacity Prediction Network

This repository is a Proof of Concept (P.O.C) to demonstrate the capabilities of Deep Learning in the public sector. The network predicts if the used capacity of a parking garage will increase or decrease in 5 minute intervals. All work here is done in combination with and on behalf of the City of Aarhus, in Aarhus Denmark.

### Data:
The data used in this model was sourced from Open Data DK, which can be found [here](https://portal.opendata.dk/dataset/parkeringshuse-i-aarhus/resource/2a82a145-0195-4081-a13c-b0e587e9b89c?view_id=c1d12524-a50b-4a5c-be21-daa741bb00bb). The data has been aggregated since March 2017 and saved on a private database. An exported variant of this database can be found under the `/data` folder. The data has been cleaned extensively through notebooks found under `/notebooks`, though there currently is no central cleaning function. Two features have been engineered:

- `future` The percentage of capacity filled 5 minutes into the future, using the next entry on the dataset.
- `target` Binary indicator of 0 if the capacity increased and 1 if the capacity decreased.

### Model:

#### Architecure:
The underlying architecture of the model is a sequential Recurrent Neural Network with _LSTM_, _Batch Normalization_, and _Dense_ layers using a _Rectified Linear_ and _Soft Max_ activation parameters and a _Sparse Categorical Crossentropy_ loss function.

#### Model Training:
The most updated model was trained with a Batch Size of 64 with 10 Epochs. The model is saved as an `.h5` file under the `/models` folder. 

#### Model Accuracy: 
The model was tested on _~51,000_ entries of unseen data and was able to predict increase/decrease at a __79.21%__ accuracy. 

### Enviroment setup:

Install dependencies with:
```pip install -r requirements.txt``` On a Mac, ensure that xcode is installed. Install Python 3 from [here](https://www.anaconda.com/distribution/).

### Questions:

All questions can go to `sharan@uw.edu`. _This model was primarily possible because of [this amazing tutorial](https://www.youtube.com/watch?v=yWkpRdpOiPY)_
