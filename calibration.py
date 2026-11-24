import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

reference_temperature = 273 #kelvin

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

print(voltages)
print(probe_temperatures)

#this is the function we are going to fit to our data
def voltage_function(probe_temperature,alpha):
    return alpha * (probe_temperature - reference_temperature) #eq 17.4 from Practical Physics 1 Handbook 2026-27

fig, ax = plt.subplots()
ax.plot(probe_temperatures, voltages,linestyle="",marker=".")
ax.set_xlabel("Temperature at probe junction (K)")
ax.set_ylabel("Voltage measured at multimeter (V)")

fitted_alpha, parameter_covariance = sp.optimize.curve_fit(voltage_function, probe_temperatures, voltages)
fitted_alpha = fitted_alpha[0] #means we get a number not a np.array
print(f"alpha = {fitted_alpha} ± {parameter_covariance}")

# line_of_best_fit_xdata = np.array()
# line_of_best_fit_ydata = line_of_best_fit_xdata * fitted_alpha

# ax.plot(line_of_best_fit_xdata,line_of_best_fit_ydata,marker='')

plt.show()