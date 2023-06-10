import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('xtick', labelsize=7) #fontsize of the x tick labels
plt.rc('ytick', labelsize=7) #fontsize of the y tick labels
import control
from ExtKalmanFilter import *
from DynamicsModel import *
from scipy.linalg import sqrtm

torch.set_default_dtype(torch.float64)



A = f_Jacobian(torch.zeros(rx, 1), torch.zeros(ru, 1)) #Linearization about 0
print(R)
Q_lqr = 0.1 * torch.diag(1.0 * torch.ones(rx,))
R_lqr = torch.diag(1.0 * torch.ones(ru,))
Klqr, _, _ = control.dlqr(A, torch.tensor([[1, 0],[0, 1]]), Q_lqr, R_lqr)
Klqr = torch.from_numpy(Klqr)
# Define model: dynamics, dynamics gradients, and noise covariances.
model = Model(stateDynamics, measurementDynamics, f_Jacobian, g_Jacobian, Q, R)
# Define information state: (mean, covariance), from an extended Kalman Fitler.
x0=torch.randn(rx, 1)
P0=torch.diag(torch.ones(rx,))
EKF = Extended_KF(x0, P0, model)
EKF.printem()


Horizon_Length = 16
x_EKF_record = torch.zeros(rx, Horizon_Length+1)
x_EKF_record[:,0] = EKF.Mean.squeeze()
x_true = torch.zeros(rx, Horizon_Length+1)
x_true[:, 0] = (EKF.Mean + torch.from_numpy(sqrtm(EKF.Covariance).real) @ torch.randn(rx, 1)).squeeze() #we might need next t-step

for k in range(Horizon_Length):
    u = -Klqr @ EKF.Mean
    true_state_Plus = stateDynamics(x_true[:, k], u) + torch.from_numpy(sqrtm(Q).real) @ torch.randn(rx, 1)
    measurement_Plus_realization = measurementDynamics(true_state_Plus, u) + torch.from_numpy(sqrtm(R).real) @ torch.randn(ry, 1)
    EKF.ApplyEKF(u, measurement_Plus_realization)
    x_true[:, k+1] = true_state_Plus.squeeze()
    x_EKF_record[:,k+1] = EKF.Mean.squeeze()

plt.plot(x_true.T, '-o', linewidth = 1)
plt.plot(x_EKF_record.T, linewidth = 1)
plt.xlabel('time-step $k$')
plt.ylabel('magnitude')
plt.legend(('$x^1_{true}$', '$x^2_{true}$', '$x^1_{estimate}$', '$x^2_{estimate}$'))
plt.savefig('Figures/EKF Rollout.PNG',bbox_inches ="tight")


    
    

