import numpy as np
import h5py

from lisatools.utils.utility import AET
from lisatools.sampling.samplingguide import MBHGuide

# from ldc.waveform.lisabeta import FastMBHB

from lisatools.pipeline.pipeline import *

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as np

    gpu_available = False

use_gpu = gpu_available

xp = cp if use_gpu else np

nwalkers = 80
ntemps = 4

nwalkers_relbin = 400
ntemps_relbin = 10

fpin = "/home/mlk667/GPU4GW/ldc/datasets/LDC1-1_MBHB_v2_FD_noiseless.hdf5"
# fpin = (
#    "/Users/michaelkatz/Research/GPU4GW/ldc/datasets/LDC1-1_MBHB_v2_FD_noiseless.hdf5"
# )
with h5py.File(fpin, "r") as f:
    grp = f["H5LISA"]["GWSources"]["MBHB-0"]
    print(list(f["H5LISA"]))

    data = f["H5LISA"]["PreProcess"]["TDIdata"][:]
    t, Xd, Yd, Zd = data[:4194304].T

"""
with h5py.File("data/mbhb-tdi.h5", "r") as f:
    grp = f["H5LISA"]["GWSources"]["MBHB-0"]
    for key in grp:
        params[key] = grp[key][()]
    print(list(f["H5LISA"]))

    data = f["H5LISA"]["PreProcess"]["TDIdata"][:]
    t, Xd, Yd, Zd = data[:4194304].T

limit = int(1e6)
with h5py.File("data/mbhb-tdi.h5", "r") as f:
    t = f['X'][:limit, 0]
    Xd = f['X'][:limit, 1]
    Yd = f['Y'][:limit, 1]
    Zd = f['Z'][:limit, 1]
"""
t -= t[0]

dt = t[1] - t[0]

T = t[-1]
fd, Xfd, Yfd, Zfd = (
    np.fft.rfftfreq(len(Xd), dt),
    np.fft.rfft(Xd) * dt,
    np.fft.rfft(Yd) * dt,
    np.fft.rfft(Zd) * dt,
)

Afd, Efd, Tfd = AET(Xfd, Yfd, Zfd)

folder = "/projects/b1095/mkatz/ldc2a/"
# sfolder = "./"
fp_search = folder + "TEST_ERYN_mode_hop_mbh_pipeline_final_NO_noise_search.h5"
fp_search_rel_bin = (
    folder + "TEST_ERYN_mode_hop_mbh_pipeline_final_NO_noise_search_rel_bin.h5"
)
fp_pe_rel_bin = (
    folder + "TEST_ERYN_mode_hop_mbh_pipeline_final_NO_noise_pe_rel_bin_testing.h5"
)

info = InfoManager(name="ldc1data", data=[Afd, Efd, Tfd], fd=fd, dt=dt, T=1 / fd[1])

mbh_search_module = MBHBase(name="initial search")
mbh_search_module.initialize_module(
    fp_search, nwalkers, ntemps, snr_stopping=100.0, search=True, use_gpu=use_gpu
)

mbh_search_rel_bin_module = MBHRelBinSearch(name="relbin search")
mbh_search_rel_bin_module.initialize_module(
    fp_search_rel_bin, nwalkers_relbin, ntemps_relbin, use_gpu=use_gpu
)

mbh_pe_rel_bin_module = MBHRelBinPE(name="relbin pe")
mbh_pe_rel_bin_module.initialize_module(
    fp_pe_rel_bin, nwalkers_relbin, ntemps_relbin, use_gpu=use_gpu
)

# info.fp_search_rel_bin = "TEST_ERYN_mbh_pipeline_final_NO_noise_search_rel_bin.h5"
module_list = [mbh_search_module, mbh_search_rel_bin_module, mbh_pe_rel_bin_module]
# module_list = [mbh_pe_rel_bin_module]

mbh_pipeline = PipelineGuide(info, module_list)

mbh_pipeline.run(verbose=True, progress=True)

exit()
fp_search = "/projects/b1095/mkatz/ldc2a/mbh_pipeline_full_NO_noise_search_3.h5"
fp_pe = "/projects/b1095/mkatz/ldc2a/mbh_pipeline_full_NO_noise_pe_4.h5"

breakpoint()
info = InfoManager(name="ldc1data", data=[Afd, Efd, Tfd], fd=fd, dt=dt, T=t[-1])

mbh_full_search_module = MBHBase(name="initial full search")
mbh_full_search_module.initialize_module(
    fp_search, nwalkers, ntemps, search=True, n_iter_stop=30, use_gpu=use_gpu
)

mbh_full_pe_module = MBHBase(name="full pe")
mbh_full_pe_module.initialize_module(
    fp_pe, nwalkers, ntemps, search=False, use_gpu=use_gpu
)

info.fp_search = "/projects/b1095/mkatz/ldc2a/mbh_pipeline_NO_noise_pe_rel_bin.h5"
# module_list = [mbh_full_search_module, mbh_full_pe_module]

breakpoint()
module_list = [mbh_pe_rel_bin_module]

mbh_pipeline = PipelineGuide(info, module_list)

mbh_pipeline.run(verbose=True)
