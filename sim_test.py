import numpy as np
import pandas as pd

real_df = pd.DataFrame(np.random.random((8087, 20)))

covariance = real_df.cov()

eig_val, eig_vec= np.linalg.eig(covariance)

z = np.random.random((20,))

x = eig_vec.dot(np.diag(eig_val ** 0.5)) * z

print(x)