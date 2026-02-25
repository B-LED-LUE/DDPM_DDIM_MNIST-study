@torch.no_grad()
def reverse_process_DDIM_5_eta_0(model, noise, device, n,steps = 5,eta=0):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps)

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_50_eta_0(model, noise, device, n,steps = 50,eta=0):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps)

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_500_eta_0(model, noise, device, n,steps = 500,eta=0):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps)
  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_5_eta_03(model, noise, device, n,steps = 5,eta=0.3):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
    )
  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_50_eta_03(model, noise, device, n,steps = 50,eta=0.3):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
    )

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_500_eta_03(model, noise, device, n,steps = 500,eta=0.3):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
    )
  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_5_eta_07(model, noise, device, n,steps = 5,eta=0.7):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
    )
  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_50_eta_07(model, noise, device, n,steps = 50,eta=0.7):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
  )

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)

@torch.no_grad()
def reverse_process_DDIM_500_eta_07(model, noise, device, n,steps = 500,eta=0.7):
  model.eval()
  steps = steps
  batch_size = n
  xt = torch.randn(batch_size,1,28,28,device = device)
  t = torch.linspace(noise.T-1, 0, steps, device=device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(xt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = model(xt, t_batch)
    alpha_bar_t = noise.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise.alphas_bars[t_end]
    pred_x0 = (xt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta*((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one-alpha_bar_t)/alpha_bar_t_minus_one)

    xt = (torch.sqrt(alpha_bar_t_minus_one)*pred_x0
    + torch.sqrt(1-alpha_bar_t_minus_one-sigma_t)*pred_eps
    + torch.sqrt(sigma_t)*eps
  )

  xt = (xt+1)/2
  xt = xt.clamp(0,1)
  show_image(xt)