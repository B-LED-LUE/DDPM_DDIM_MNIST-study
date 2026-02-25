import torch
import torch.nn as nn
import math
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)
#------------------------------------------------------------------------------
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

setting = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5,))
])
all_data = datasets.MNIST(root = '/data', train = True, download = True, transform = setting)
five = (all_data.targets == 5).nonzero(as_tuple=True)[0]
five_data = DataLoader(Subset(all_data, five), batch_size = 128, shuffle = True)

#------------------------------------------------------------------------------
class DDPM(nn.Module):
  def __init__(self, T=1000 ,beta_start=1e-4, beta_end=0.02, device=device):
    super().__init__()
    self.T = T
    self.betas = torch.linspace(beta_start, beta_end, T, device=device)
    self.alphas = 1 - self.betas
    self.alphas_bars = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_bars = torch.sqrt(self.alphas_bars)
    self.sqrt_one_minus_alphas_bars = torch.sqrt(1-self.alphas_bars)

  def forward(self, x0 ,t):
    noise = torch.randn_like(x0)
    sqrt_alphas_bars = self.sqrt_alphas_bars[t].view(-1,1,1,1)
    sqrt_one_minus_alphas_bars = self.sqrt_one_minus_alphas_bars[t].view(-1,1,1,1)
    return sqrt_alphas_bars * x0 + sqrt_one_minus_alphas_bars * noise, noise
#------------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.mlp = nn.Sequential(
        nn.Linear(dim, dim*4),
        nn.SiLU(),
        nn.Linear(dim*4, dim)
    )
  def forward(self, t):
    device = t.device
    half_dim = self.dim // 2

    x = math.log(10000)/(half_dim - 1)
    x = torch.exp(torch.arange(half_dim, device = device)*-x)
    x = t[:,None]*x[None,:]
    x = torch.cat((x.sin(),x.cos()), dim = -1)
    x = self.mlp(x)
    return x
#----------------------------------------------------------------------------------
class ResBlock(nn.Module):
  def __init__(self, in_ch, out_ch, time_dim):
    super().__init__()

    self.time_mlp = nn.Sequential(
        nn.SiLU(),
        nn.Linear(time_dim, out_ch)
    )

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding = 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(out_ch, out_ch, 3, padding = 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

    self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

  def forward(self, x, t_emb):
    #x:[B,in_ch, H, W]
    #t_emb: time vector [B, timd_dim]

    h = self.conv1(x)
    t_emb_refined = self.time_mlp(t_emb).view(-1, h.shape[1],1,1)
    h = h + t_emb_refined
    h = self.conv2(h)
    return h + self.shortcut(x)
#----------------------------------------------------------------------------------


class UNet(nn.Module):
  def __init__(self, in_ch = 1, time_dim = 128):
    super().__init__()
    self.time_mlp = TimeEmbedding(time_dim)
    self.init_conv = nn.Conv2d(in_ch, 64, 3,padding=1)

    self.down1 = ResBlock(64, 128, time_dim)
    self.pool1 = nn.Conv2d(128, 128, 3, stride = 2, padding = 1)

    self.down2 = ResBlock(128, 256, time_dim)
    self.pool2 = nn.Conv2d(256,256,3, stride = 2, padding = 1)

    self.mid1 = ResBlock(256, 512, time_dim)
    self.mid2 = ResBlock(512, 256, time_dim)

    self.up1 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
    self.up_res1 = ResBlock(256 + 256, 128, time_dim)

    self.up2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
    self.up_res2 = ResBlock(128 + 128, 64, time_dim)

    self.out_conv = nn.Conv2d(64, in_ch, 1)

  def forward(self, x, t):
    t_emb = self.time_mlp(t)
    x1 = self.init_conv(x)
    x2 = self.down1(x1, t_emb)
    x3 = self.pool1(x2)
    x4 = self.down2(x3, t_emb)
    x5 = self.pool2(x4)

    x6 = self.mid1(x5, t_emb)
    x7 = self.mid2(x6, t_emb)

    x = self.up1(x7)
    x = torch.cat([x,x4],dim=1)
    x = self.up_res1(x,t_emb)
    x = self.up2(x)
    x = torch.cat([x,x2],dim=1)
    x = self.up_res2(x,t_emb)
    return self.out_conv(x)
#-----------------------------------------------------------------------------
def forward_process(model,data, noise, epochs=40,device= device):
  model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
  print("start")
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x0,_ in data:
      x0 = x0.to(device)
      t = torch.randint(0, noise.T, (x0.shape[0],), device = device)
      xt, noise_t = noise(x0, t)

      pred = model(xt, t)
      loss = F.mse_loss(pred, noise_t)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    print(f"Epoch[{epoch+1}/{epochs}]| Loss: {loss.item():.6f}")
#-------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def show_image(images, n=64):
    imgs = images.detach().cpu().numpy().clip(0, 1)

    rows, cols = 4, 16
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axes.flatten()

    for i in range(n):
        if i < len(imgs) and i < len(axes):
            axes[i].imshow(imgs[i].squeeze(), cmap='gray')
            axes[i].axis('off')


    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.show()
    plt.close()
#-----------------------------------------------------------------------------------------------------
model = UNet(in_ch = 1, time_dim = 128).to(device)
noise = DDPM(T=1000, beta_start = 1e-4, beta_end=0.02, device = device)
data = five_data
forward_process(model, data, noise,epochs=30,device= device)

reverse_process_DDIM_5_eta_0(model, noise, device, n,steps = 5,eta=0)
print("5 eta=0")
reverse_process_DDIM_50_eta_0(model, noise, device, n,steps = 50,eta=0)
print("50 eta=0")
reverse_process_DDIM_500_eta_0(model, noise, device, n,steps = 500,eta=0)
print("500 eta=0")
reverse_process_DDIM_5_eta_03(model, noise, device, n,steps = 5,eta=0.3)
print("5 eta=03")
reverse_process_DDIM_50_eta_03(model, noise, device, n,steps = 50,eta=0.3)
print("50 eta=03")
reverse_process_DDIM_500_eta_03(model, noise, device, n,steps = 500,eta=0.3)
print("500 eta=03")
reverse_process_DDIM_5_eta_07(model, noise, device, n,steps = 5,eta=0.7)
print("5 eta=07")
reverse_process_DDIM_50_eta_07(model, noise, device, n,steps = 50,eta=0.7)
print("50 eta=07")
reverse_process_DDIM_500_eta_07(model, noise, device, n,steps = 500,eta=0.7)
print("500 eta=07")
reverse_process_DDPM(model, noise, device, n)
print("reverse_process_DDPM")
reverse_process_DDPM_C(model, noise, device, n)
print("reverse_process_DDPM_C")
reverse_process_DDPM_500(model, noise, device, n)
print("reverse_process_DDPM_500")
reverse_process_DDPM_500_C(model, noise, device, n)
print("reverse_process_DDPM_500_C")

