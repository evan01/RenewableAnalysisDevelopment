# now we should have all the information we need to do the real statistical analysis


def plotTimeSeries(self, data):
    data.plot()
    plt.savefig("./plots/originalSeries.png")


def plotHistogram(self, data):
    data.hist()
    plt.savefig("./plots/histogram.png")


def plotRampVCapacity(self, data):
    """
    This function plots a number of figures showing the relationships between ramp and capacity
    :param data:
    :return:
    """
    # opts = self.options
    # First thing to do is to get rid of the na vals, they seem to pop up often
    data.dropna(inplace=True, how='any')

    x = data[self.label]
    y = data['ramp']

    x = x.as_matrix()
    y = y.as_matrix()

    # There are multiple different kinds of plots for ramp and capacity
    sns.jointplot(x=self.label, y='ramp', data=data)  # Standard scatter
    sns.jointplot(x=self.label, y='ramp', data=data, kind="kde", ylim={-80, 80}, xlim={0, 1500},
                  color='r')  # A kind of heatmap
    sns.jointplot(x=self.label, y='ramp', data=data, kind='hex', ylim={-80, 80}, xlim={0, 1500},
                  color='r')  # Hex bin plot

    # Try some parametrization
    parametrized = sns.jointplot(x=self.label, y='ramp', data=data)
    parametrized.plot_marginals(sns.distplot)

    # Try to draw hills
    g = sns.JointGrid(x=self.label, y='ramp', data=data, ylim=(-80, 80), xlim=(0, 1000), size=5, ratio=2)
    g = g.plot_joint(sns.kdeplot, cmap="Reds_d")
    g = g.plot_marginals(sns.kdeplot, color='r', shade=True)

    # Try to draw a simple kde plot...
    sns.kdeplot(x, y, ylim={-80, 80})  # A hill like contour plot

    sns.plt.show()
    print("done")


def getBivariateDistribution(self, data, GRID):
    """
    This will get the bivariate distribution of the data set and plot the output
    :param data:
    :param GRID:
    :return:
    """
    # Might be worthwhile to remove outliers... hmm kmeans might help with this
    data.dropna(inplace=True, how='any')
    x = data[self.label].as_matrix()
    y = data['ramp'].as_matrix()

    # Params to find using data
    Expectation_x = x.mean()
    Expectation_y = y.mean()

    sig_x = int(x.var() ** .5)
    sig_y = int(y.var() ** .5)

    # This is to give to the pdf function
    print("Applying the binning, meshgrid function")
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y

    print("Aquiring normal distribution")
    Z = BIV_NORM.bivariate_normal(X, Y, sig_x, sig_y, Expectation_x, Expectation_y)

    print("Plot the distribution")

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


    # Let scipy.stats do the multivariate normal distribution heavy lifting, pass in covariance matrix


def plotStatistics(self):
    data = self.stats['data']
    self.getBivariateDistribution(data, 60)

    print("Done plotting all the statistics")
