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

#using non-linear least squares method from Scipy

probe_temperatures = list()
for (i, voltage) in enumerate(voltages):
    probe_temperature = (voltage/fitted_alpha) + reference_temperature
    probe_temperatures.append(probe_temperature)

probe_temperatures = np.array(probe_temperatures)

temperature_differences = list()
for j in range(len(probe_temperatures)):
    temperature_differences.append(probe_temperatures[j]-environment_temperature)

temperature_differences = np.array(temperature_differences)

def newtons_cooling_law(time, thermal_constant):
    theta = temperature_differences[0] * np.exp(-(thermal_constant) * time)
    return theta

fitted_thermal_constant_lnls, paramater_covariance = sp.optimize.curve_fit(newtons_cooling_law,times, temperature_differences)
print(fitted_thermal_constant_lnls)

plt.show()