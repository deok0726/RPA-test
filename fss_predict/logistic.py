import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 데이터 불러오기
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data = breast_cancer.data, columns = breast_cancer.feature_names)
df = df.iloc[:, :10]
df["label"] = breast_cancer.target
df.columns = [ col.replace(" ", "_") for col in df.columns]
