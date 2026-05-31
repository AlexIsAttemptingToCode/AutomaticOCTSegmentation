"""
This module provides functions for displaying images using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_decision_boundary(clf, X, y, 
                           X_test=None, y_test=None, 
                           n_steps=1000, data=None, ax=None,
                           x1lim=None, x2lim=None,
                           legend="grouped"):
    """
    Plot the decision boundaries of a classifier with two or three features.
    As a student, you don't need to understand the details of this function.
    
    Args:
        clf: The classifier to plot.
        X: The features of the dataset.
        y: The labels of the dataset.
        X_test: (optional) The features of the test dataset
        y_test: (optional) The labels of the test dataset
        n_steps: Parameter controlling the resolution of the plot.
        ax: The axis to plot on. If None, a new figure is created.
        data: Data structure provided by sklearn.datasets.load_iris().
    """
    assert isinstance(X, pd.DataFrame)
    if ax is None:
        _, ax = plt.subplots()
        
    x1, x2 = X.iloc[:, 0], X.iloc[:, 1]
    if x1lim is not None:
        x1_min, x1_max = x1lim
    else:
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
    if x2lim is not None:
        x2_min, x2_max = x2lim
    else:
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
    
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, n_steps),
        np.linspace(x2_min, x2_max, n_steps)
    )
    zz = clf.predict(pd.DataFrame(np.c_[xx1.ravel(), xx2.ravel()],
                                  columns=X.columns))
    zz = zz.reshape(xx1.shape)
    
    n_classes = len(np.unique(y))
    assert n_classes <= 3, "This function only supports up to 3 classes."
    
    blue = PALETTE[0]
    red = PALETTE[1]
    green = PALETTE[2]
    colors = [red, blue, green][:n_classes]
    from .colors import color_transitions
    cmap = color_transitions(*colors,
                             n_steps=n_classes, 
                             as_cmap=True)
    #cmap = plt.cm.RdYlBu

    cf = ax.contourf(xx1, xx2, zz, cmap=cmap, alpha=0.2)
    cf.set_edgecolors("face")
    cf.set_linewidths(0.5)
    
    handles = []
    for i, color in zip(range(n_classes), colors):
        group = []
        label = data.target_names[i] if data is not None else ("Class %d" % i)
        h = ax.scatter(
            x1[y == i],
            x2[y == i],
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.25,
            s=20,
        )
        group.append(h)
        if X_test is not None and y_test is not None:
            label = ((data.target_names[i] + " (test)") if data is not None 
                    else ("Class %d (test)" % i))
            label = None
            h = ax.scatter(
                X_test.iloc[:, 0][y_test == i],
                X_test.iloc[:, 1][y_test == i],
                color=color,
                label=label,
                edgecolor="black",
                linewidth=0.25,
                s=20,
                marker="^",
            )
            group.append(h)
        handles.append(tuple(group))
    ax.set_xlabel(x1.name, fontweight="bold")
    ax.set_ylabel(x2.name, fontweight="bold")
    
    # Add legend with different markers for test data, if present.
    # https://matplotlib.org/stable/users/explain/axes/legend_guide.html
    if legend == "grouped":
        from matplotlib.legend_handler import HandlerTuple
        suffix = ""
        if data is not None:
            labels = data.target_names
        else:
             labels = ["Class %d" % i for i in range(n_classes)]
            
        if X_test is not None and y_test is not None:
            suffix = " (train/test)"
        ax.legend(handles=handles, 
                labels=[name+suffix for name in labels],
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fontsize="small")
    elif legend == "simple":
        ax.legend()
    elif legend in ("none", False, None):
        ax.legend().remove()
    else:
        raise ValueError("Invalid value for 'legend'.")
