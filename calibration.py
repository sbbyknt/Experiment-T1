import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.style.use("ggplot")

reference_temperature = 273

datasets = {
            "calibration_data_water_bath_75C.csv": 348,
            "calibration_data_dry_ice.csv": 195,
            "calibration_data_ice.csv": 273,
            }

voltages = list()
probe_temperatures = list()

for file in datasets:
    data = np.loadtxt(file,delimiter=",",skiprows=1)
    voltages.append(data[-1][-1] * 1e-3)
    probe_temperatures.append(datasets[file])

#eq 17.4 from Practical Physics 1 Handbook 2026-27, what we are fitting to data
def voltage_function(probe_temperature,alpha):
    return alpha * (probe_temperature - reference_temperature) 

fitted_alpha, parameter_covariance = sp.optimize.curve_fit(voltage_function, probe_temperatures, voltages)
fitted_alpha = fitted_alpha[0] #means we get a number not a np.array
print(f"\u03b1 = {fitted_alpha} ± {parameter_covariance}")

#initialises graph
fig, ax = plt.subplots()
ax.set_xlabel("Temperature at probe junction (K)")
ax.set_ylabel("Voltage measured at multimeter (V)")

ax.plot(probe_temperatures, voltages,linestyle="",marker=".",label="Data")

#line of best fit handling
lobf_xdata = np.array([np.min(probe_temperatures), np.max(probe_temperatures)])
lobf_ydata = fitted_alpha * (lobf_xdata - reference_temperature)

ax.plot(lobf_xdata, lobf_ydata, linestyle=":", marker=" ",label="Linear fit",color="r")
#visually can clearly see that linear fit to calibration data suitable

plt.legend()
plt.show()