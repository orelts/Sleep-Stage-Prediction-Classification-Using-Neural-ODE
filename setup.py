from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Project purpose is to use Neural ODE for sleeping stage identification and prediction of the incoming measurement. Neural ODEs are neural network models which generalize standard layer to layer propagation to continuous depth models. In our project, the data is a time series of EEG measurements during people sleeping time. We want to feed this data to the model and see the performance of our model in classifying and predicting sleeping stages.',
    author='Orel Tsioni',
    license='',
)
