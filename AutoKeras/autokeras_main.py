from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive
from multi_class.autokeras_wrapper import AutoKerasWrapper
import autokeras as ak
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

if __name__ == '__main__':
    data = pd.read_csv("./Pizza.csv")
    enc = OrdinalEncoder(dtype=np.int32)
    enc.fit(data[["brand"]])
    data[["brand"]] = enc.transform(data[["brand"]])
    X = data.drop(["brand"], axis=1)
    y = data["brand"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    automl = ak.StructuredDataClassifier(overwrite=True,
                                            max_trials=3,
                                            seed=42)
    automl.fit(x=X_train, y=y_train, epochs=1)

    wrapper = AutoKerasWrapper(automl)

    dashboard = ExplainerDashboard(ClassifierExplainer(wrapper, X_test, y_test, labels=enc.categories_[0].tolist()))


    dashboard.save_html("./dashboard.html")
    dashboard.explainer.dump("./dashboard.dill")

    dashboard = ExplainerDashboard(ClassifierExplainer.from_file("./dashboard.dill"))
    dashboard.run(8044)