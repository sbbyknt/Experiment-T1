import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from calibration import fitted_alpha, reference_temperature

plt.style.use("ggplot")
fig, ax = plt.subplots(1,3)

environment_temperature = 19.6 + 273 #also in Kelvin

data = np.loadtxt("experimental_data.csv",delimiter=",",skiprows=1)
times = data[:,0]
voltages = data[:,1] * 1e-3

ax[0].plot(times,voltages,linestyle=" ", marker=".")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Voltage (V)")

probe_temperatures = list()
for (i, voltage) in enumerate(voltages):
    probe_temperature = -(voltage/fitted_alpha) + reference_temperature
    probe_temperatures.append(probe_temperature)

probe_temperatures = np.array(probe_temperatures)

temperature_differences = list()
for j in range(len(probe_temperatures)):
    temperature_differences.append(probe_temperatures[j]-environment_temperature)

temperature_differences = np.array(temperature_differences)

ax[1].plot(times, temperature_differences,linestyle=" ", marker=".")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$\Theta$")

#we now need to find β. There are two ways of doing this globally:
# 1. non-linear least squares method from Scipy.optimize
# 2. linear regression, which we could implement by hand (but will use np.polyfit)

#using non-linear least squares method from Scipy

def newtons_cooling_law(time, thermal_constant):
    theta = temperature_differences[0] * np.exp(-(thermal_constant) * time)
    return theta

fitted_thermal_constant_nlls, parameter_covariance_nlls = sp.optimize.curve_fit(newtons_cooling_law,times, temperature_differences)
std_fitted_thermal_constant_nlls = np.sqrt(parameter_covariance_nlls[0][0])

#using linear regression by linearising function

# given theta = theta_0 e^{-beta t}, taking logs of both sides gives
# log(theta) = log(theta_0 e^{-beta t}) = log(theta_0) + (-beta * t).
#
# we therefore want to plot t against log(theta), which will give us a gradient of -beta

log_temperature_differences = np.log(-temperature_differences)

ax[2].plot(times, log_temperature_differences, marker=".", linestyle=" ")

(minus_beta, ln_theta_0), parameter_covariance_lr  = np.polyfit(times, log_temperature_differences,1,cov=True)
fitted_thermal_constant_lr = float(minus_beta / -1)
std_fitted_thermal_constant_lr = np.sqrt(parameter_covariance_lr[0][0])

ax[2].set_xlabel("$\log{t}$ (s)")
ax[2].set_ylabel("$\log{\Theta}$")

print(f"β given by scipy.optimize non-linear least squares method is {fitted_thermal_constant_nlls[0]} ± {std_fitted_thermal_constant_nlls}. β given by linear regression is {fitted_thermal_constant_lr} ± {std_fitted_thermal_constant_lr}")
#thermal constants are very close 

#we will now try and fit these to their respective curves

lobf_xdata = np.linspace(np.min(times), np.max(times), 1000)
#y data as given with thermal constant from nlls scipy.optimize method
lobf_ydata_nlls = newtons_cooling_law(lobf_xdata, fitted_thermal_constant_nlls)
#y data as given with lr from np.polyfit method
lobf_ydata_lr = newtons_cooling_law(lobf_xdata,fitted_thermal_constant_lr)

#these plots basically sit on top of each other
# ax[1].plot(lobf_xdata, lobf_ydata_nlls, linestyle=":", marker=" ", color="r", label="Fit of $\Theta = \Theta_0 e^{\beta t}$ using $\beta$ from `scipy.optimize` method")
ax[1].plot(lobf_xdata,lobf_ydata_lr,linestyle=":", marker=" ", color="b", label="Fit of $\Theta = \Theta_0 e^{\beta t}$ using $\beta$ from linear regression method")

lobf_log_ydata_lr = list()
for j in range(len(lobf_xdata)):
    lobf_log_ydata_lr.append(-1 * fitted_thermal_constant_lr * lobf_xdata[j] + np.log(-temperature_differences[0]))
print(lobf_log_ydata_lr)
ax[2].plot(lobf_xdata, lobf_log_ydata_lr, linestyle=":", marker=" ", color="b")

plt.legend()
plt.show()