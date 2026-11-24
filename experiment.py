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
# ax.plot(times,voltages,linestyle=" ",marker=".",color="r")
# ax.set_xlabel("Time ($s$)")
# ax.set_ylabel("Voltage ($V$)")

# plt.show()

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

#using linear regression by linearising function

'''
Given theta = theta_0 e^{-beta t}, taking logs of both sides gives
log(theta) = log(theta_0 e^{-beta t}) = log(theta_0) + (-beta * t).

We therefore want to plot t against log(theta), which will give us a gradient of -beta
'''
log_temperature_differences = list()
for i in range(len(temperature_differences)):
    if temperature_differences[i] > 0:
        log_temperature_differences.append(np.log(temperature_differences[i]))

log_temperature_differences = np.array(log_temperature_differences)


times_lr = list()
for j in range(len(log_temperature_differences)):
    times_lr.append(times[j])

times_lr = np.array(times_lr)

ax.plot(times_lr, log_temperature_differences, marker=".", linestyle=" ")
ax.set_xlabel("$\log{t}$ (s)")
ax.set_ylabel("$\log{\Theta}$ (K)")
plt.show()