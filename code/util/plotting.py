import matplotlib.pyplot as plt
import numpy as np

from util.prepare_data import f

def setup(size):
    plt.rc('text', usetex=True)
    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = size
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

def visualize_dataset(x, y):
    
    grid = np.linspace(0, 1, 100)
    fvals = f(grid)
    
    plt.figure(figsize=(12, 8))
    setup((12, 8))
    
    plt.title("Toy dataset - Visualization")
    plt.plot(grid, fvals, "k--", label="True function")
    plt.plot(x, y, "ro", label="Observed data points")
    plt.grid()
    plt.ylabel("f(x)")
    plt.xlabel("x-axis")
    plt.xlim(0, 1)
    plt.legend()

    plt.show()

def visualize_predictions(X_train, y_train, X_test, mu, var):
    plt.figure(figsize=(12, 8))
    setup((12, 8))

    plt.title("DNGO - Visualization")
    plt.grid()
    
    plt.plot(X_train, y_train, "ro", label="Observed data points")
    plt.plot(X_test[:, 0], f(X_test[:,0]), "k--", label="True function")
    
    plt.plot(X_test[:, 0], mu, "blue", label="Predicted mean function")
    plt.fill_between(X_test[:, 0], mu + np.sqrt(var), mu - np.sqrt(var), color="orange", alpha=0.8)
    plt.fill_between(X_test[:, 0], mu + 2 * np.sqrt(var), mu - 2 * np.sqrt(var), color="orange", alpha=0.6)
    plt.fill_between(X_test[:, 0], mu + 3 * np.sqrt(var), mu - 3 * np.sqrt(var), color="orange", alpha=0.4)
    plt.xlim(0, 1)
    plt.xlabel(r"Input $x$")
    plt.ylabel(r"Output $f(x)$")
    plt.legend()
    plt.show()

def loss_plot(num_epochs, loss_arr):
    plt.figure(figsize=(12, 8))
    setup((12, 8))
    plt.grid()

    plt.title("Training loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.plot([i for i in range(num_epochs)], loss_arr, label="Training loss", marker='o')
    plt.legend()
    plt.show()

def visualize_EI(X_train, y_train, X_test, mu, var, EI):
    plt.figure(figsize=(15, 14))
    setup((15, 13))

    fg, axes = plt.subplots(2, 1)
    ax1, ax2 = axes

    # Plotting predictions
    ax1.grid()
    
    ax1.plot(X_train, y_train, "ro", label="Observed data points")
    ax1.plot(X_test[:, 0], f(X_test[:,0]), "k--", label="True function")
    
    ax1.plot(X_test[:, 0], mu, "blue", label="Predicted mean function")
    ax1.fill_between(X_test[:, 0], mu + np.sqrt(var), mu - np.sqrt(var), color="orange", alpha=0.8)
    ax1.fill_between(X_test[:, 0], mu + 2 * np.sqrt(var), mu - 2 * np.sqrt(var), color="orange", alpha=0.6)
    ax1.fill_between(X_test[:, 0], mu + 3 * np.sqrt(var), mu - 3 * np.sqrt(var), color="orange", alpha=0.4)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(r"Input $x$")
    ax1.set_ylabel(r"Output $f(x)$")
    ax1.legend()

    # Plotting Aquisition function
    ax2.grid()

    ax2.plot(X_test[:,0], EI, label="Expected Improvement")
    ax2.fill_between(X_test[:,0], [0.0 for i in range(X_test.shape[0])], [y[0] for y in EI], color="skyblue", alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"Input $x$")
    ax2.set_ylabel(r"$EI(x)$")
    ax2.legend()

    plt.show()


