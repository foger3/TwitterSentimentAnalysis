def cool_add(a, b):
    return(a + b)


def ploty():
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 20, 100)
    plt.plot(x, np.sin(x))
    plt.show()