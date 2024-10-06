# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 00:12:59 2022

@author: hassa
"""
import numpy as np
import scipy as sp
import scipy.special
import numpy as np
import math
import matplotlib.pyplot as plt


import warnings

from matplotlib import pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.baseclasses import SchwarzschildEccentric, Pn5AAK, ParallelModuleBase, KerrCircular
from few.waveform import FastSchwarzschildEccentricFlux, AAKWaveformBase, Pn5AAKWaveform, Disk_Kerr_AAK_Waveform, SlowSchwarzschildEccentricFlux,RelativisticKerrCircularFlux

from few.summation.aakwave import AAKSummation, KerrAAKSummation
from few.utils.utility import get_mismatch, get_ode_function_options, get_p_at_t, get_separatrix, get_fundamental_frequencies
from few.waveform import   GenerateEMRIWaveform
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.waveform import SchwarzschildEccentricWaveformBase, KerrRelativisticWaveformBase
from few.utils.baseclasses import SchwarzschildEccentric
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.interp2dcubicspline import Interp2DAmplitude, Interp2DAmplitude_Kerr
from few.summation.interpolatedmodesum import InterpolatedModeSum, CubicSplineInterpolant, InterpolatedModeSum_Kerr
from few.summation.directmodesum import DirectModeSum
from few.utils.ylm import GetYlms
from few.amplitude.romannet import RomanAmplitude
from lisatools.diagnostic import snr

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False




use_gpu = gpu_available #change this to True for gpu
import multiprocessing

num_threads = 2#multiprocessing.cpu_count()


# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e7),  # dense stepping trajectories
        "func":"SchwarzEccFlux"
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e4),  # this must be >= batch_size
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu  # GPU is available for this type of summation
}



Kerr_inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e7),  # dense stepping trajectories
        "func": "Relativistic_Kerr_Circ_Flux"
    }




insp_kwargs_AAK_rel = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e7),
    "func":"Relativistic_Kerr_Circ_Flux"
    }

sum_kwargs_AAK = {
    "use_gpu": gpu_available,  # GPU is availabel for this type of summation
    "pad_output": False,
}

insp_kwargs_AAK = {
    "err": 1e-12,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e7),
    "func":"pn5"
    }

gen_wave_AAK = GenerateEMRIWaveform(
    AAKWaveformBase, # Define the base waveform
    EMRIInspiral, # Define the trajectory
    KerrAAKSummation, # Define the summation for the amplitudes
    inspiral_kwargs=insp_kwargs_AAK,
    sum_kwargs=sum_kwargs_AAK,
    use_gpu=use_gpu,
    frame="detector"
    )


# gen_wave_AAK_rel = GenerateEMRIWaveform(
# AAKWaveformBase, # Define the base waveform
# EMRIInspiral, # Define the trajectory
# KerrAAKSummation, # Define the summation for the amplitudes
# inspiral_kwargs=insp_kwargs_AAK_rel,
# sum_kwargs=sum_kwargs_AAK,
# use_gpu=use_gpu,
# # return_list=True,
# frame="detector"
# )


gen_wave_Kerr = GenerateEMRIWaveform(
KerrRelativisticWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
Interp2DAmplitude_Kerr, # Define the interpolation for the amplitudes
InterpolatedModeSum_Kerr, # Define the type of summation
inspiral_kwargs=Kerr_inspiral_kwargs,
sum_kwargs=sum_kwargs,
amplitude_kwargs=amplitude_kwargs,
use_gpu=use_gpu,
frame='detector'
)


traj_l30_Has = EMRIInspiral(func="Relativistic_Kerr_Circ_Flux") # my flux and lmax = 30
axion_traj = EMRIInspiral(func="Axion_Relativistic_Kerr_Circ_Flux")

num_threads





dt = 1.0
p0 = 10.0
e0 = 0.0
a0 = 0.6
x0 = 1.0
Y0 = 1.0
qK = 0.0  # polar spin angle
phiK = 0.0  # azimuthal viewing angle
qS = 0.0  # polar sky angle
phiS = 0.0  # azimuthal viewing angle
dist = 1.0  # distance
Phi_theta0 = 0.0
Phi_r0 = 0.0




T = 4.0 # years
eta = 1e-5
# M = 4e5
mu = 20
M = mu/eta

print("M:", M, "mu:", mu)   


traj_args = [M, mu, a0, e0, Y0]
p0 = get_p_at_t(traj_l30_Has, T, traj_args,index_of_p=3,index_of_a=2,index_of_e=4,index_of_x=5,
                bounds=[get_separatrix(a0,e0,x0)+0.1, 43.0])
print("p0:", p0, "mu:", mu)




# from the mcmc injection params for Axions
qK = Pi/6.0  # polar spin angle
phiK = Pi/5.0  # azimuthal viewing angle
qS = Pi/4.0  # polar sky angle
phiS = Pi/3.0  # azimuthal viewing angle
Phi_phi0 = Pi/3.0





lmax = 30
specific_modes = []
for l in range(2,lmax+1):
    for m in range(0,l+1):
        specific_modes += [(l,m,0)]


specific_modes22 = [(2,2,0)]



Kerr_wave_gen = gen_wave_Kerr(M, mu, a0, p0, e0, Y0,dist, qS, phiS, qK, phiK,Phi_phi0,Phi_theta0,Phi_r0,  T=T, dt=dt, mode_selection=specific_modes)
Kerr_wave_gen_nparr = Kerr_wave_gen.get()

Kerr_wave_gen22 = gen_wave_Kerr(M, mu, a0, p0, e0, Y0,dist, qS, phiS, qK, phiK,Phi_phi0,Phi_theta0,Phi_r0,  T=T, dt=dt, mode_selection=specific_modes22)
Kerr_wave_gen_nparr22 = Kerr_wave_gen22.get()




tkerr = np.arange(0, len(Kerr_wave_gen_nparr))*dt/YRSID_SI

print(gpu_available)

psd_kwargs_4y = {'includewd': 4.0}  # Pass includewd=4.0 as a keyword argument
psd_kwargs_1y = {'includewd': 1.0}  # Pass includewd=4.0 as a keyword argument

check_snr_Kerr = snr(Kerr_wave_gen.real, dt=dt,PSD = "cornish_lisa_psd" ,use_gpu=gpu_available) 
check_snr_Kerr2 = snr(Kerr_wave_gen.real, dt=dt,PSD = "lisasens", PSD_kwargs=psd_kwargs_1y, use_gpu=gpu_available) 
check_snr_Kerr3 = snr(Kerr_wave_gen.real, dt=dt,PSD = "lisasens", PSD_kwargs=psd_kwargs_4y, use_gpu=gpu_available) 
check_snr_Kerr_22 = snr(Kerr_wave_gen22.real, dt=dt,PSD = "lisasens", PSD_kwargs=psd_kwargs_4y, use_gpu=gpu_available) 

print("SNR for Kerr with Cornish psd:", check_snr_Kerr)
print("SNR for Kerr with SciRDv1 psd with 1 year WD:", check_snr_Kerr2)
print("SNR for Kerr with SciRDv1 psd with 4 year WD:", check_snr_Kerr3) 
print("SNR for Kerr with SciRDv1 psd with 4 year WD for 22 mode:", check_snr_Kerr_22)

from lisatools.sensitivity import *

# breakpoint()

# Tobs = T
# N_obs = int(Tobs * YRSID_SI / dt)
ffth_Kerr = xp.fft.rfft(Kerr_wave_gen)
power_Kerr = xp.abs(ffth_Kerr)**2
f_arr  = xp.fft.rfftfreq(len(Kerr_wave_gen),dt)
# f_arr = f_arr.get()
power_Kerr = power_Kerr[f_arr!=0.0]
f_arr = f_arr[f_arr!=0.0]



######## For the 2,2 mode waveform
ffth_kerr22 = xp.fft.rfft(Kerr_wave_gen22)
power_Kerr22 = xp.abs(ffth_kerr22)**2
f_arr22  = xp.fft.rfftfreq(len(Kerr_wave_gen22),dt)
power_Kerr22 = power_Kerr22[f_arr22!=0.0]
f_arr22 = f_arr22[f_arr22!=0.0]



zero_idx = np.where(f_arr == 0.0)
# print(zero_idx, f_arr[zero_idx])






sens1 = get_sensitivity(f_arr, sens_fn="cornish_lisa_psd")
sens4 = get_sensitivity(f_arr, sens_fn="lisasens")
# sens2 = get_sensitivity(f_arr, sens_fn="noisepsd_AE", includewd = T)
# year = 365.25 * 24.0 * 3600.0
sens3 = get_sensitivity(f_arr.get(), sens_fn="lisasens", includewd = 4.0)
sens5 = get_sensitivity(f_arr.get(), sens_fn="lisasens", includewd = 1.0)


plt.figure(figsize=(10, 8))
plt.loglog(f_arr.get(), sens1.get(), label="Cornish Lisa PSD")
# plt.loglog(f_arr, sens2, label="AE Noise PSD")
plt.loglog(f_arr.get(), sens3,'--', label="LISA Sensitivity (SciRDv1) with 4 years WD")
plt.loglog(f_arr.get(), sens4.get(),'--', label="LISA Sensitivity (SciRDv1) without WD")
plt.loglog(f_arr.get(), sens5,'-.', label="LISA Sensitivity (SciRDv1) with 1 year WD")
plt.loglog(f_arr.get(), power_Kerr.get(), label="Kerr Waveform", rasterized = True)
plt.loglog(f_arr22.get(), power_Kerr22.get(), label="Kerr Waveform 2,2 mode", rasterized = True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity")
plt.xlim(1e-5, 1e-1)
plt.ylim(1e-43, 1e-32)
plt.legend()
plt.savefig("sensitivity_curves.png")




# Sgal1= WDconfusionAE(f_arr, duration=4.0, model="SciRDv1", use_gpu=use_gpu)
# Sgal2 = GalConf(f_arr, Tobs=4.0*year,  use_gpu=use_gpu)
# plt.figure()
# plt.loglog(f_arr.get(), Sgal1.get(), label="WD Confusion AE")
# plt.loglog(f_arr.get(), Sgal2.get(), label="Galactic Confusion")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Sensitivity")
# plt.legend()
# plt.savefig("sensitivity_curves_galactic.png")