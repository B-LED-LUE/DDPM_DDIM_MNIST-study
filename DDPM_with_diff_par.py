@torch.no_grad()
def reverse_process_DDPM(model, noise, device, n):
  model.eval()
  xt = torch.randn(n, 1, 28, 28, device = device)
  for t in reversed(range(noise.T)):
    t_batch = torch.full((n,), t, device = device).long()
    pred = model(xt, t_batch)
    alpha = noise.alphas[t]
    alpha_bar_t = noise.alphas_bars[t]
     
    beta = noise.betas[t]
    beta_tilde=(1-noise.alphas_bars[t-1])/(1-alpha_bar_t)*beta if t > 0 else 0

    z = torch.randn_like(xt) if t > 0 else 0
    xt = ((1 / torch.sqrt(alpha))*(xt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +math.sqrt(beta_tilde)*z)

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDPM_C(model, noise, device, n):
  model.eval()
  xt = torch.randn(n, 1, 28, 28, device = device)
  for t in reversed(range(noise.T)):
    t_batch = torch.full((n,), t, device = device).long()
    pred = model(xt, t_batch)
    alpha = noise.alphas[t]
    alpha_bar_t = noise.alphas_bars[t]
     
    beta = noise.betas[t]
    beta_tilde=(1-noise.alphas_bars[t-1])/(1-alpha_bar_t)*beta if t > 0 else 0

    z = torch.randn_like(xt) if t > 0 else 0
    xt = ((1 / torch.sqrt(alpha))*(xt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +math.sqrt(beta_tilde)*z)
    for _ in range(3):
      xt = LD_Corrector_for_DDPM(model, noise, t, xt, device, n)


  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDPM_500_C(model, noise, device, n):
  model.eval()
  xt = torch.randn(n, 1, 28, 28, device = device)
  for t in range(noise.T-1, -1, -2):
    t_batch = torch.full((n,), t, device = device).long()
    pred = model(xt, t_batch)
    alpha = noise.alphas[t]
    alpha_bar_t = noise.alphas_bars[t]
     
    beta = noise.betas[t]
    beta_tilde=(1-noise.alphas_bars[t-1])/(1-alpha_bar_t)*beta if t > 0 else 0
    z = torch.randn_like(xt) if t > 0 else 0
    xt = ((1 / torch.sqrt(alpha))*(xt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +math.sqrt(beta_tilde)*z)
    for _ in range(3):
      xt = LD_Corrector_for_DDPM(model, noise, t, xt, device, n)

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDPM_500(model, noise, device, n):
  model.eval()
  xt = torch.randn(n, 1, 28, 28, device = device)
  for t in range(noise.T-1, -1, -2):
    t_batch = torch.full((n,), t, device = device).long()
    pred = model(xt, t_batch)
    alpha = noise.alphas[t]
    alpha_bar_t = noise.alphas_bars[t]
     
    beta = noise.betas[t]
    beta_tilde=(1-noise.alphas_bars[t-1])/(1-alpha_bar_t)*beta if t > 0 else 0
    z = torch.randn_like(xt) if t > 0 else 0
    xt = ((1 / torch.sqrt(alpha))*(xt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +math.sqrt(beta_tilde)*z)


  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)