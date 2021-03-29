
def plot_spectre():
    fig = None
    scale = False
    if isinstance(plot, str):
        scale = plot
        plot = True
    if plot is True:
        fig, plot = plt.subplots()

    N = len(spe)//2+1

    if split:
        W = 0.8
        w = W/spe.shape[1]
        for i in range(spe.shape[1]):
            y = spe[:len(spe)//2+1, i]
            if split == 'normed':
                y = y/y.sum()
            else:
                y = y / spe[:N].sum(axis=-1).mean()
            x = np.arange(len(y))
            plot.bar(x+w/2-W/2+i*w, y, width=w, bottom=0.001, zorder=10)
    else:
        y = spe[:N] / spe[:len(spe)//2+1].sum()
        x = np.arange(len(y))
        plot.bar(x, y, width=.8, bottom=0.001, zorder=10, color='gray')

    plot.spines['top'].set_visible(False)
    plot.spines['right'].set_visible(False)
    plot.spines['left'].set_visible(False)

    plot.set_xticks(np.arange(0, N, 1))
    xlabels = ['Equivariant','$2\pi$', '$\pi$'][:min(3,N)]
    xlabels += ['$\dfrac{2\pi}{%i}$'%_ if _%2 else '$\dfrac{\pi}{%i}$'%(_//2)
                for _ in range(3,N)]
    plot.set_xticklabels(xlabels)

    plot.set_ylabel('Polar Spectral Power Density')
    plot.set_ylim([0.001,1])
    plot.set_yticks([.25,.5,.75, 1])
    plot.set_yticklabels(['25%','50%','75%', '100%'])
    plot.yaxis.grid()
    if scale:
        plot.set_yscale(scale)
    plot.grid(which='minor', color='#bbbbbb', linestyle='-', linewidth=1, zorder=1)

    if fig is not None:
        fig.show()