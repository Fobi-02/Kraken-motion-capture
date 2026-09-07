import pandas as pd
import numpy as np

df = pd.DataFrame({
    "id": [1, 2, 3],
    "valore": [1.5, 2.7, 3.2]
})

df["matrice"] = None  # crea la colonna

df.at[0, "matrice"] = np.array([
    [1, 2],
    [3, 4]
])

print(df)