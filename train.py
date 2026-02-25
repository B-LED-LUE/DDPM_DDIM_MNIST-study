model = UNet(in_ch = 1, time_dim = 128).to(device)
noise = DDPM(T=1000, beta_start = 1e-4, beta_end=0.02, device = device)
data = five_data
forward_process(model, data, noise,epochs=30,device= device)