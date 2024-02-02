import numpy as np
import torch

# TODO: unify to either numpy or torch.

def GBC(f_s: np.ndarray, y: np.ndarray, current_labels):

    perclass = dict.fromkeys(current_labels)
    for c in current_labels:
        f_s_c = f_s[y == c, :].detach()

        N_c = f_s_c.shape[0]
        mu_c = sum(f_s_c) / N_c
        mu_c = mu_c.unsqueeze(1)

        cov_c = sum((f_s_c[i, :] - mu_c) @ (f_s_c[i, :] - mu_c).T for i in range(N_c)) / (N_c-1)

        perclass[c] = [mu_c, cov_c]

    gbc = 0
    for i in range(len(perclass.keys())):
        for j in range(i+1, len(perclass.keys())):
            bc = torch.exp(-bhattacharyya_dist(perclass[i], perclass[j]))

            print(bc)
            gbc -= bc

    return gbc

def bhattacharyya_dist(c_i, c_j):

    mu_c_i, mu_c_j = c_i[0], c_j[0]
    cov_c_i, cov_c_j = c_i[1], c_j[1]

    cov = (cov_c_i + cov_c_j) / 2

    cov += torch.eye(cov.shape[0]) * 1e-3

    D_b = (mu_c_i - mu_c_j).T @ torch.inverse(cov) @ (mu_c_i - mu_c_j) / 8 + np.log(torch.det(cov) / np.sqrt(torch.det(cov_c_i) * torch.det(cov_c_j)))
    
    print("Sigma: ", cov, "\nSigma inverse", torch.inverse(cov))
    print("Determinant: ", torch.det(cov))

    print((np.sqrt(torch.det(cov_c_i) * torch.det(cov_c_j))))
    
    return D_b
