import joblib
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
study =joblib.load('SC4001E0.pkl')
df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
print(optuna.visualization.is_available())
print(df.head(20))
plot_optimization_history(study).show()
print("hi")