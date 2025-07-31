import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
    'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
    'Label': ['Blue'] * 8 + ['Red'] * 8
}

df = pd.DataFrame(data)
print(df)
X1 = df['x1']
X2 = df['x2']

colors = df['Label'].map({'Blue': 'blue', 'Red': 'red'})

plt.figure(figsize=(6, 6))
plt.scatter(X1, X2, c=colors)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Original 2D Points")
plt.show()

def Transform(x1, x2):
    return np.array([x1**2, np.sqrt(2)*x1*x2, x2**2])

transformed_points = np.array([Transform(X1, X2) for X1, X2 in zip(X1, X2)])
print(transformed_points)

fig =   plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(transformed_points[:,0], transformed_points[:,1], transformed_points[:, 2], c=colors)
ax.set_xlabel("X1^2")
ax.set_ylabel("X2^2")
ax.set_zlabel("sqrt(2) * X1 * X2")
ax.set_title("Transformed 3D Points")
plt.show()

