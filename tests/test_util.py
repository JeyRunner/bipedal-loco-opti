import os
import matplotlib.pyplot as plt
import numpy as np
from casadi import Function


def save_fig(test_object, name_suffix=None):
	fig_path = 'figures/'
	os.makedirs(fig_path, exist_ok=True)

	if name_suffix is not None:
		name_suffix = "_" + name_suffix
	else:
		name_suffix = ""
	plt.savefig(fig_path + test_object._testMethodName + name_suffix + '.png')







def casadi_eval(x):
    f_func = Function('f',  # func name
                      [],  # input symbols
                      [x],  # output symbol expression
                      [],  # input symbol names
                      ['y']  # output symbol names
                      )

    # print('f_func', f_func)
    # print('f_func', f_func())
    # print('f_func', f_func()['y'])
    return f_func()['y']


def to_np(x) -> np.ndarray:
    return np.array(casadi_eval(x))