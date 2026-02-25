@torch.no_grad()
def LD_Corrector_for_DDPM(model, noise,t, xt,device, n):
  model.eval()
  no_zero_time = max(0, t-1)
  t_batch = torch.full((n,), no_zero_time, device = device).long()

  alpha_bar = noise.alphas_bars[no_zero_time]
  score = -model(xt,t_batch)/(torch.sqrt(1-alpha_bar))
  beta_now = noise.betas[no_zero_time]
  eta = 0.01 * beta_now
  noise = torch.randn_like(xt) if t > 0  else 0
  xt = xt + eta * score + torch.sqrt(2 * eta) * noise
  return xt
