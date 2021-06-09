import torch
import nonexp

import matplotlib.pyplot as plt

tr_function = nonexp.Transmittance.apply

# Initialize some example extinction "sigmat" and transmittance mode "gamma"
res = 128
t = torch.linspace(0, 1, res, dtype=torch.float64).cuda()
sigmat = torch.abs(torch.sin(10 * t) + torch.cos(3*t)+1) / res
gamma = torch.clamp(0.5 * (torch.sin(3 * t) + 0.5), 0, 1)

# The op assumes input arrays to have a 2D shape
gamma = gamma[None, :]
sigmat = sigmat[None, :]

tr = tr_function(sigmat, gamma)

# Plot the evaluation of the model
plt.figure()
plt.plot(t.cpu(), sigmat[0].cpu(), label='sigmat')
plt.plot(t.cpu(), gamma[0].cpu(), label='gamma')
plt.legend()
plt.figure()
plt.plot(t.cpu(), tr[0].cpu())
plt.show()

# Check the gradients for correctness
sigmat.requires_grad = True
gamma.requires_grad = True
torch.autograd.gradcheck(tr_function, [sigmat, gamma])