import numpy as np
from numpy.linalg import inv
from numpy.random import dirichlet, randint, multivariate_normal
import pandas as pd
import scipy as sp
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import sys


# File input
file_ip = sys.argv[1]
cluster_op = sys.argv[2]
posterior_means_op = sys.argv[3]


# Creates database for data
df = pd.read_csv(file_ip)
X = df.to_numpy()

 # Hyperparameters
k = 2
DATAPOINTS = len(df)
BURN_IN    = 30
Sigma = np.eye(2)
V_0   = 10*Sigma


# Helper Functions
def display(labels): 
    df["labels"] = labels
    cmap = plt.get_cmap('tab10')
    scatter = plt.scatter(df["1.211"], df["2.552"],c=df["labels"], cmap=cmap)
    plt.colorbar(scatter, ticks=range(k), label = "Class label")
    plt.show()

def write_cluster_membership(file, Data, labels):
    Data["labels"] = labels
    df.to_csv(file, index=False)

def write_posterior_means(file, Data):
    np.savetxt(file, Data, delimiter=",", fmt="%.4f")



def gibbs_sampler(T=100):
# Initializations
    alpha        = np.ones(k)
    z_cache      = randint(0,k, DATAPOINTS) # array of datapoints class assignments
    N_cache      = np.bincount(z_cache, minlength=k)
    X_cache      = np.array([X[z_cache==i].mean(axis=0) for i in range(k)])
    V_inv_cache  = inv(V_0)+ N_cache[:, None, None]*Sigma #For broadcasting, we now have (k,2,2)+(k,2,2)*(2,2)
    m_cache      = inv(V_inv_cache)@(N_cache[:, None]*X_cache)[:,:,None]  # This is a (k,2,2) @ (2,2) which broadcasts to a (k,2,2)
    m_cache      = m_cache.squeeze(-1)
    mu_cache     = np.array([multivariate_normal(mean=m_cache[i], cov=inv(V_inv_cache[i])) for i in range(k)])
    pi_cache     = dirichlet(alpha+N_cache)

    stored_z = []
    stored_mu = []

    for t in range(T):
        for i in range(DATAPOINTS):
            z = z_cache[i]
            old_N = N_cache[z]
            new_N = N_cache[z] - 1

            N_cache[z] = new_N
            X_cache[z] = (X_cache[z] - X[i]/old_N) *(old_N/new_N)

            probs = np.empty(k)
            for j in range(k):
                probs[j] = np.log(pi_cache[j]) + sp.stats.multivariate_normal.logpdf(X[i], mean= mu_cache[j], cov=Sigma)
            probs -= logsumexp(probs)
            probs = np.exp(probs)


            z = np.random.choice(k, p=probs)
            z_cache[i] = z
            old_N = N_cache[z]
            new_N = N_cache[z]+1

            N_cache[z] = new_N
            X_cache[z] = (X_cache[z] + (X[i]/old_N)) *(old_N/new_N)

            V_inv_cache  = inv(V_0)+ N_cache[:, None, None]*Sigma #For broadcasting, we now have (k,2,2)+(k,2,2)*(2,2)
            m_cache  = inv(V_inv_cache)@(N_cache[:, None]*X_cache)[:,:,None]  # This is a (k,2,2) @ (2,2) which broadcasts to a (k,2,2)
            m_cache = m_cache.squeeze(-1)
            mu_cache = np.array([multivariate_normal(mean=m_cache[i], cov=inv(V_inv_cache[i])) for i in range(k)])
            pi_cache = dirichlet(alpha+N_cache)
        if (t > BURN_IN):
            # stored_z.append(z_cache.copy())
            # stored_mu.append(mu_cache.copy())
            # Since we are just outputting the last iteration values, it doesn't make sense to keep copy of this data
            pass
    return z_cache, mu_cache

#
def main():
    z, mu =gibbs_sampler()
    # display(z)
    write_cluster_membership(cluster_op, df, z)
    write_posterior_means(posterior_means_op, mu)



if __name__ == "__main__":
    main()


