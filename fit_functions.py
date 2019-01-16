import numpy as np
from scipy.optimize import curve_fit

for sex in ["male", "female"]:
  reference_table = np.genfromtxt(sex + ".csv", delimiter="\t")
  param_names = "abcdefghijklmnopqrstuvw"
  ages = np.array([19, 23, 28, 33, 38, 43, 48, 53, 58])
  measures = np.array([2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 35])
  while measures[-1] < 60:
    measures = np.concatenate([measures, measures[-1:] + 2])
    sl = reference_table[:,-1] + 0.5
    reference_table = np.concatenate([reference_table, sl.reshape(sl.shape[0], 1)], axis=1)
  input_tuples = np.array([[(a, m) for m in measures] for a in ages])

  def caliper_function(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
    age = x[:, 0]
    measure = x[:, 1]
    return a + b * age + c * measure + d * age**2 + e * measure**2 + f * age**3 + g * measure**3 + h * age**4 + i * measure**4 + j * age**5 + k * measure**5 + l * age**6 + m * measure**6 

  params, pcov = curve_fit(caliper_function, input_tuples.reshape(input_tuples.shape[0] * input_tuples.shape[1], 2), reference_table.flatten()) 
  print("== optimizing function for {} table ==".format(sex))
  print("\t".join([" {}: {}".format(param_names[i], p) for i, p in enumerate(params)]))
  error = 0
  for i, age in enumerate(ages):
    print("\t".join(["{:0.1f}".format(caliper_function(np.array([[age, measure]]), *params)[0]) for j, measure in enumerate(measures)]))
    error += sum([abs(reference_table[i, j] - caliper_function(np.array([[age, measure]]), *params)[0]) for j, measure in enumerate(measures)])
  print("{} error: {:0.8f}".format(sex, error))

