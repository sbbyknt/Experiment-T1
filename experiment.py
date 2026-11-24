import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")
import scipy as sp
from calibration import fitted_alpha, reference_temperature

environment_temperature = 19.6 + 273 #also in Kelvin

data = np.loadtxt("experimental_data.csv",delimiter=",",skiprows=1)
times = data[:,0]
voltages = data[:,1]

fig, ax = plt.subplots()
ax.plot(times,voltages,linestyle=" ",marker=".",color="r")
ax.set_xlabel("Time ($s$)")
ax.set_ylabel("Voltage ($V$)")

# probe_temperatures = list()
# for (i, voltage) in enumerate(voltages):
#     probe_temperature = (voltage/fitted_alpha) + reference_temperature
#     probe_temperatures.append(probe_temperature)

# probe_temperatures = np.array(probe_temperatures)

# temperature_differences = list()
# for j in range(len(probe_temperatures)):
#     temperature_differences.append(probe_temperatures[j]-environment_temperature)

# temperature_differences = np.array(temperature_differences)

# def newtons_cooling_law(time, thermal_constant):
#     theta = temperature_differences[0] * np.exp(-(thermal_constant) * time)
#     return theta

# fitted_thermal_constant, paramater_covariance = sp.optimize.curve_fit(newtons_cooling_law,times, temperature_differences)
# print(fitted_thermal_constant)

# ax.plot(times, temperature_differences,linestyle=" ",marker=".",color="r")
# ax.set_xlabel("Time ($s$)")
# ax.set_ylabel("Temperature difference ($\Delta T$)")

# #handles line of best fit
# line_of_best_fit_xdata = np.linspace(times[0],times[-1],1000)
# line_of_best_fit_ydata = [newtons_cooling_law(time, fitted_thermal_constant) for time in line_of_best_fit_xdata]
# ax.plot(line_of_best_fit_xdata, line_of_best_fit_ydata, marker=" ")

plt.show()