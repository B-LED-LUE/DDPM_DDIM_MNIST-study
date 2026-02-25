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
