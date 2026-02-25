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
