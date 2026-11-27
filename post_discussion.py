import numpy as np
from calibration import fitted_alpha, reference_temperature, error_in_fitted_alpha

environment_temperature = 19.6 + 273 #also in Kelvin

def newtons_cooling_law(time, thermal_constant):
    theta = temperature_differences[0] * np.exp(-(thermal_constant) * time)
    return theta

data = np.loadtxt("experimental_data.csv",delimiter=",",skiprows=1)
times = data[:,0]
voltages = data[:,1] * 1e-3

cutoff = 6
voltages = voltages[cutoff:]
times = times[cutoff:]


probe_temperatures = list()
for (i, voltage) in enumerate(voltages):
    probe_temperature = -(voltage/fitted_alpha) + reference_temperature
    probe_temperatures.append(probe_temperature)

probe_temperatures = np.array(probe_temperatures)

temperature_differences = list()
for j in range(len(probe_temperatures)):
    temperature_differences.append(probe_temperatures[j]-environment_temperature)

temperature_differences = np.array(temperature_differences)

log_temperature_differences = np.log(-temperature_differences)

(minus_beta, ln_theta_0), parameter_covariance_lr  = np.polyfit(times, log_temperature_differences,1,cov=True)
fitted_thermal_constant_lr = float(minus_beta / -1)
std_fitted_thermal_constant_lr = np.sqrt(parameter_covariance_lr[0][0])

#reduced chi-squared test 
expected_ydata = newtons_cooling_law(times, fitted_thermal_constant_lr)
residuals = temperature_differences - expected_ydata

std_V = 0.01E-3
std_theta = np.sqrt(((-1 * std_V)/fitted_alpha)**2 + ((voltages * error_in_fitted_alpha)/fitted_alpha ** 2)**2)

dof = len(temperature_differences) - 1 

chi_squared = 0 
for k in range(len(residuals)):
    chi_squared += (residuals[k] / std_theta[k]) ** 2

reduced_chi_squared = chi_squared / dof 

print(f"β = {fitted_thermal_constant_lr} ± {std_fitted_thermal_constant_lr}")
print(f"Reduced chi-squared test value: {reduced_chi_squared}")