import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import kSZclass as ksz
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

#%%
# Set-up

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

k_arr = np.logspace(-3, 1, 128)
lk_arr = np.log(k_arr)
a_arr = np.linspace(0.1, 1, 32)

log10M = np.linspace(11, 15, 1000)
M = 10**log10M

H_arr = cosmo['h'] * ccl.h_over_h0(cosmo, a_arr) / ccl.physical_constants.CLIGHT_HMPC
f_arr = cosmo.growth_rate(a_arr)

aHf_arr = a_arr * H_arr * f_arr

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 14})

#%%
# Overdensities

#%%

# Halo mass definition
hmd_200m = ccl.halos.MassDef200m

# Concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# Mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

# Halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# Matter overdensity 
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

# Galaxy overdensity
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1, bg_0=2.04)

# Halo model integral calculator
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)

# Gas density profile
profile_parameters = {"lMc": 14.0, "beta": 0.6, "eta_b": 0.5, "A_star": 0.03}
pGas = hp.HaloProfileDensityHE_withFT(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
pGas.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=300, plaw_fourier=-2.0)

#%%
# Cross-correlations

#%%

# Matter-matter
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-matter
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy
pk_eg = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy one-halo term only
pk_eg_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)

# Electron-galaxy two-halo term only
pk_eg_2h = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_1h=False, get_2h=True)

# Electron-electron
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pGas, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-galaxy
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy with satellites
log10M = np.linspace(11, 15, 1000)
M_vals = 10**log10M
n_M = nM(cosmo, M_vals, 1/(1+0.55))
b_M = bM(cosmo, M_vals, 1/(1+0.55))
sat = ksz.Satellites(M_vals, M0=1e11, M1=3e12, M_min=1.74e12, nM=n_M)
M_mean_sat = sat.mean_halo_mass()

pG_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(1.74e12), log10M0_0=np.log10(1e11), log10M1_0=np.log10(3e12), alpha_0=1.1)
pk_eg_sat = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG_sat, lk_arr=lk_arr, a_arr=a_arr)


pk_dict = {
    'mm': {'full': pk_mm},
    'eg': {'full': pk_eg, '1h': pk_eg_1h, '2h': pk_eg_2h, 'sat': pk_eg_sat},
    'gm': {'full': pk_gm},
    'em': {'full': pk_em},
    'gg': {'full': pk_gg},
    'ee': {'full': pk_ee},
    }

#%%
# Calculate 3D power spectra

#%%

# Initialise a kSZ object using kSZclass.py
kSZ = ksz.kSZclass(cosmo, k_arr, a_arr, pk_dict)

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = np.array([a * H * f])
a_index = 0

# Contribution from the trispectrum term (transverse)
pkt_tri_4h = np.array([np.float64(-7.203464751484216e-06), np.float64(-5.7067538883843824e-06), np.float64(-5.472405724350699e-06), np.float64(-5.213591532965721e-06), np.float64(-4.930317290848859e-06), np.float64(-4.621826040498523e-06), np.float64(-4.2864768892701526e-06), np.float64(-3.9237587965323565e-06), np.float64(-3.5294837371217343e-06), np.float64(-3.1085450954019245e-06), np.float64(-2.6565055696857724e-06), np.float64(-2.1735537466343777e-06), np.float64(-1.6580129530679487e-06), np.float64(-1.1104378597367306e-06), np.float64(-5.304723830944471e-07), np.float64(8.114294594088589e-08), np.float64(7.248948247555021e-07), np.float64(1.3994768879296172e-06), np.float64(2.1114729973716868e-06), np.float64(2.8409486762551362e-06), np.float64(3.5919305139210453e-06), np.float64(4.360910724431833e-06), np.float64(5.141872643766284e-06), np.float64(5.936746261148497e-06), np.float64(6.7233802907227564e-06), np.float64(7.503727950933803e-06), np.float64(8.2672064727978e-06), np.float64(9.0018190584575e-06), np.float64(9.687649272246492e-06), np.float64(1.0311650445379507e-05), np.float64(1.0869692350013253e-05), np.float64(1.1339269601798368e-05), np.float64(1.1732274192901684e-05), np.float64(1.2033896822262906e-05), np.float64(1.2237261559783755e-05), np.float64(1.2311972224950943e-05), np.float64(1.2267709309094856e-05), np.float64(1.2106368812800721e-05), np.float64(1.1848664863480247e-05), np.float64(1.1522474935430247e-05), np.float64(1.1150032266526389e-05), np.float64(1.0716046278531043e-05), np.float64(1.0189183956111537e-05), np.float64(9.618953924853044e-06), np.float64(9.011500238648682e-06), np.float64(8.442379355291473e-06), np.float64(7.935229392130558e-06), np.float64(7.501197858561677e-06), np.float64(7.051761738986145e-06), np.float64(6.529784269417755e-06), np.float64(5.956856926937488e-06), np.float64(5.3489427693849224e-06), np.float64(4.780106314561275e-06), np.float64(4.3003328443425506e-06), np.float64(3.898483904610495e-06), np.float64(3.4401563666821127e-06), np.float64(2.928115157591513e-06), np.float64(2.4415451266193986e-06), np.float64(2.03415952581533e-06), np.float64(1.7436676438662532e-06), np.float64(1.6032339750002785e-06), np.float64(1.5001074364640365e-06), np.float64(1.2844291311263217e-06), np.float64(1.025211522173256e-06), np.float64(7.995628696153652e-07), np.float64(6.279522048630457e-07), np.float64(5.173774714186038e-07), np.float64(4.843792381143517e-07), np.float64(4.367455533077555e-07), np.float64(3.510238473354614e-07), np.float64(2.795711092007742e-07), np.float64(2.174418799487736e-07), np.float64(1.6245283102816533e-07), np.float64(1.3256104661993902e-07), np.float64(1.3026786787845144e-07), np.float64(1.142681184966123e-07), np.float64(9.108867062371734e-08), np.float64(7.036111352544435e-08), np.float64(5.438451494278402e-08), np.float64(4.089008016584974e-08), np.float64(3.6324717227650926e-08), np.float64(3.5987129523338355e-08), np.float64(2.9631256784120384e-08), np.float64(2.3731131156044944e-08), np.float64(1.8804162920698525e-08), np.float64(1.4689372681941453e-08), np.float64(1.1702792104794811e-08), np.float64(1.0701376846646039e-08), np.float64(9.827232282933552e-09), np.float64(7.914374120738021e-09), np.float64(6.339805372081012e-09), np.float64(5.112750646895981e-09), np.float64(3.980647874713243e-09), np.float64(3.220786461438484e-09), np.float64(2.8689224259018308e-09), np.float64(2.4111353624962773e-09), np.float64(1.9042312046741277e-09), np.float64(1.5302445074591695e-09), np.float64(1.1997664187882932e-09), np.float64(9.098584436554879e-10), np.float64(7.09965846622967e-10), np.float64(6.078859095116526e-10), np.float64(4.989954135593855e-10), np.float64(4.0263087846965003e-10), np.float64(3.184019465863243e-10), np.float64(2.352578213863686e-10), np.float64(1.6711259209368221e-10), np.float64(1.2432094427632433e-10), np.float64(1.0321062522683034e-10), np.float64(8.520713542949185e-11), np.float64(6.676457417223497e-11), np.float64(4.9509043116267656e-11), np.float64(3.468802159657135e-11), np.float64(2.2860068028898634e-11), np.float64(1.6790072281080083e-11), np.float64(1.4290029725680141e-11), np.float64(1.1681871536928509e-11), np.float64(8.976920690218291e-12), np.float64(6.5989710691829395e-12), np.float64(4.24793824282761e-12), np.float64(2.6446039008488885e-12), np.float64(2.0593476598495574e-12), np.float64(1.8369468844626364e-12), np.float64(1.4852942647976853e-12), np.float64(1.1730206752582734e-12), np.float64(9.342043143091147e-13), np.float64(7.55134933057561e-13), np.float64(1.0972818008493456e-12)])

# Contribution from the bispectra terms (longitudinal)
pkp_bi_3h = [np.float64(-0.006337066416847899), np.float64(-0.005852496761354781), np.float64(-0.0054023830653361655), np.float64(-0.004984265956211715), np.float64(-0.004595878464290308), np.float64(-0.004235124929955716), np.float64(-0.003900062063402897), np.float64(-0.0035888776654999515), np.float64(-0.0032998933397754403), np.float64(-0.003031573836930002), np.float64(-0.0027825107841028475), np.float64(-0.00255136055018977), np.float64(-0.0023368932749326865), np.float64(-0.0021379699869753773), np.float64(-0.0019535549130753394), np.float64(-0.0017826736626439476), np.float64(-0.0016244082851770228), np.float64(-0.0014779294341735777), np.float64(-0.0013424368641044742), np.float64(-0.001217217145671068), np.float64(-0.0011015842806720627), np.float64(-0.000994913460283803), np.float64(-0.0008965972480107049), np.float64(-0.0008060919017244257), np.float64(-0.0007228809519182822), np.float64(-0.0006464789381676782), np.float64(-0.0005764467110907674), np.float64(-0.0005123496671509018), np.float64(-0.0004538040961681322), np.float64(-0.00040043432780978027), np.float64(-0.0003518885330237997), np.float64(-0.00030783811444010654), np.float64(-0.00026798672026184757), np.float64(-0.00023205103495455056), np.float64(-0.00019976694686785535), np.float64(-0.00017088921424068763), np.float64(-0.0001451785902109607), np.float64(-0.00012241530304326367), np.float64(-0.0001023970042568231), np.float64(-8.492315860514638e-05), np.float64(-6.979562226545317e-05), np.float64(-5.683609167798079e-05), np.float64(-4.5859816191975784e-05), np.float64(-3.667059377475006e-05), np.float64(-2.9087786565042826e-05), np.float64(-2.2908217432092142e-05), np.float64(-1.7941386718465742e-05), np.float64(-1.4014172686706989e-05), np.float64(-1.0951179233032698e-05), np.float64(-8.556774858586505e-06), np.float64(-6.662597759118101e-06), np.float64(-5.172279539986745e-06), np.float64(-3.960637733247668e-06), np.float64(-2.925392521690167e-06), np.float64(-2.0462480791470932e-06), np.float64(-1.3145473466473082e-06), np.float64(-7.051583635996751e-07), np.float64(-2.469086230840711e-07), np.float64(3.774838250317965e-08), np.float64(2.0447604629577487e-07), np.float64(2.932326762647896e-07), np.float64(3.211827479163536e-07), np.float64(3.499838235341862e-07), np.float64(4.0907890627501567e-07), np.float64(4.3243482345993676e-07), np.float64(4.092332618241118e-07), np.float64(3.792221969392345e-07), np.float64(3.4227713266109634e-07), np.float64(3.0477557237063783e-07), np.float64(2.853060308800746e-07), np.float64(2.5705862210690025e-07), np.float64(2.1547527556885267e-07), np.float64(1.9124628816034601e-07), np.float64(1.7259748538246712e-07), np.float64(1.4321876225243148e-07), np.float64(1.1937879835087083e-07), np.float64(1.0634897188839192e-07), np.float64(9.037424151058638e-08), np.float64(7.245192479922602e-08), np.float64(6.173880635303959e-08), np.float64(5.1869372788760236e-08), np.float64(4.1706082795755746e-08), np.float64(3.449401880617693e-08), np.float64(2.8673771968966446e-08), np.float64(2.27395498962539e-08), np.float64(1.8235893901054208e-08), np.float64(1.5310577318855704e-08), np.float64(1.2231888781469094e-08), np.float64(9.516423426752905e-09), np.float64(7.928256929992488e-09), np.float64(6.3032882238806186e-09), np.float64(4.790737704573187e-09), np.float64(3.878207081553374e-09), np.float64(3.1235788493107374e-09), np.float64(2.385827760169271e-09), np.float64(1.8801702087094738e-09), np.float64(1.52490365822347e-09), np.float64(1.1535157444797025e-09), np.float64(8.786374293526769e-10), np.float64(7.116956283721441e-10), np.float64(5.434940160242576e-10), np.float64(4.0201758853408674e-10), np.float64(3.2242145456200236e-10), np.float64(2.488937476818616e-10), np.float64(1.812144144709231e-10), np.float64(1.4186456724896756e-10), np.float64(1.1039832193169229e-10), np.float64(7.987236847495633e-11), np.float64(6.035345464617576e-11), np.float64(4.7301721978054756e-11), np.float64(3.442439652173977e-11), np.float64(2.507086025259668e-11), np.float64(1.9656538447011287e-11), np.float64(1.4429847172576613e-11), np.float64(1.018410128784351e-11), np.float64(7.86912471823557e-12), np.float64(5.837391673211188e-12), np.float64(4.0542372147840746e-12), np.float64(3.0460142784396055e-12), np.float64(2.2809423055019832e-12), np.float64(1.5772832286283015e-12), np.float64(1.131267519332282e-12), np.float64(8.451306161663436e-13), np.float64(5.809622715509903e-13), np.float64(3.9056504656836125e-13), np.float64(2.7133776415394365e-13), np.float64(1.1948710958482496e-13), np.float64(-6.704358252196954e-12)]
pkp_bi_1h = [np.float64(9.773578660680927e-11), np.float64(9.773578393668929e-11), np.float64(9.773578084678202e-11), np.float64(9.773577727648276e-11), np.float64(9.773577314782007e-11), np.float64(9.773576837468658e-11), np.float64(9.773576285509978e-11), np.float64(9.773575647207965e-11), np.float64(9.773574909378824e-11), np.float64(9.773574056037566e-11), np.float64(9.773573069293917e-11), np.float64(9.77357192824853e-11), np.float64(9.773570609080594e-11), np.float64(9.773569083853109e-11), np.float64(9.773567320394803e-11), np.float64(9.773565281835143e-11), np.float64(9.773562925769538e-11), np.float64(9.77356020330989e-11), np.float64(9.773557057716652e-11), np.float64(9.773553423443456e-11), np.float64(9.773549225475785e-11), np.float64(9.773544376843536e-11), np.float64(9.773538776915576e-11), np.float64(9.773532309652056e-11), np.float64(9.773524841078534e-11), np.float64(9.773516216519045e-11), np.float64(9.773506257259473e-11), np.float64(9.773494756188842e-11), np.float64(9.773481473013038e-11), np.float64(9.773466129838986e-11), np.float64(9.773448405468355e-11), np.float64(9.773427929337636e-11), np.float64(9.773404273491315e-11), np.float64(9.773376943646526e-11), np.float64(9.77334536866326e-11), np.float64(9.773308891988525e-11), np.float64(9.773266759302857e-11), np.float64(9.773218098598262e-11), np.float64(9.773161903412106e-11), np.float64(9.773097012711953e-11), np.float64(9.773022085333034e-11), np.float64(9.772935574314576e-11), np.float64(9.772835692140793e-11), np.float64(9.77272036901713e-11), np.float64(9.772587207860456e-11), np.float64(9.77243343724032e-11), np.float64(9.772255865988627e-11), np.float64(9.772050813932134e-11), np.float64(9.77181401692333e-11), np.float64(9.771540518027644e-11), np.float64(9.771224573925362e-11), np.float64(9.770859534984693e-11), np.float64(9.770437708671887e-11), np.float64(9.769950207828912e-11), np.float64(9.769386776438437e-11), np.float64(9.768735539938202e-11), np.float64(9.76798281365461e-11), np.float64(9.76711282222604e-11), np.float64(9.766107403893002e-11), np.float64(9.764945641300774e-11), np.float64(9.763603208976934e-11), np.float64(9.762051984650565e-11), np.float64(9.760259500689283e-11), np.float64(9.758188336382385e-11), np.float64(9.755795368703479e-11), np.float64(9.75303071533749e-11), np.float64(9.749836576952577e-11), np.float64(9.746146749899188e-11), np.float64(9.741884843536826e-11), np.float64(9.736962578205463e-11), np.float64(9.731277806795546e-11), np.float64(9.724713141772641e-11), np.float64(9.717133374744416e-11), np.float64(9.708383297751097e-11), np.float64(9.698284540373067e-11), np.float64(9.686632435357252e-11), np.float64(9.673193707453667e-11), np.float64(9.657702836398728e-11), np.float64(9.63985414926771e-11), np.float64(9.619297264836416e-11), np.float64(9.595634324016843e-11), np.float64(9.568414824670342e-11), np.float64(9.537129423059273e-11), np.float64(9.501204771532465e-11), np.float64(9.459994267113421e-11), np.float64(9.412770668481355e-11), np.float64(9.358728257659535e-11), np.float64(9.296974222904274e-11), np.float64(9.226531794286305e-11), np.float64(9.146334127877231e-11), np.float64(9.055233097568574e-11), np.float64(8.952009844464144e-11), np.float64(8.835385572254963e-11), np.float64(8.704038446455185e-11), np.float64(8.55663115567116e-11), np.float64(8.391844496122508e-11), np.float64(8.20842543746485e-11), np.float64(8.005241633173914e-11), np.float64(7.781321236441428e-11), np.float64(7.53592876713674e-11), np.float64(7.268619599300351e-11), np.float64(6.979287298475218e-11), np.float64(6.668274740812781e-11), np.float64(6.33639435631773e-11), np.float64(5.985008369323401e-11), np.float64(5.616090844440782e-11), np.float64(5.232303826191622e-11), np.float64(4.837040418909615e-11), np.float64(4.434405483788396e-11), np.float64(4.029134632513913e-11), np.float64(3.6264299872504e-11), np.float64(3.2316977288655135e-11), np.float64(2.850309277455315e-11), np.float64(2.487437474510688e-11), np.float64(2.1478803664978657e-11), np.float64(1.8359244084378224e-11), np.float64(1.5550691401049598e-11), np.float64(1.3077247136306234e-11), np.float64(1.0950696082927328e-11), np.float64(9.170871453844028e-12), np.float64(7.725512161025191e-12), np.float64(6.588078736041449e-12), np.float64(5.715384763457016e-12), np.float64(5.047958792583987e-12), np.float64(4.513376654945337e-12), np.float64(4.0332288859963265e-12), np.float64(3.5364402131176268e-12), np.float64(2.9803256025094216e-12)]

# Contribution from the velocity term (longitudinal)
P_mm = aHf[0]**2 * pk_mm(k_arr, a) / ((2 * np.pi)**3 * k_arr**2)

# Galaxy-kSZ cross-correlation (transverse)
pkt_gk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg) for k in k_arr])
pkt_gk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkt_gk = pkt_gk1 + pkt_gk2 + pkt_tri_4h

# Galaxy-kSZ cross-correlation (longitudinal)
pkp_gk1 = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg) for k in k_arr])
pkp_gk2 = np.array([kSZ.P_par_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkp_gk = pkp_gk1 + pkp_gk2 + pkp_bi_1h + pkp_bi_3h

# Galaxy-galaxy auto-correlation (transverse)
pkt_gg1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_gg) for k in k_arr])
pkt_gg2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_gm) for k in k_arr])
pkt_gg = pkt_gg1 + pkt_gg2

# kSZ-kSZ auto-correlation (transverse)
pkt_kk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_ee) for k in k_arr])
pkt_kk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_em, pk_em) for k in k_arr])
pkt_kk = pkt_kk1 + pkt_kk2

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
pkt_gk1_1h = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_1h) for k in k_arr])

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
pkt_gk1_2h = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_2h) for k in k_arr])

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
pkp_gk1_1h = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg_1h) for k in k_arr])

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
pkp_gk1_2h = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg_2h) for k in k_arr])

# Galaxy-kSZ cross-correlation (transverse) with satellites
pkt_gk1_sat = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_sat) for k in k_arr])

# Compare M_mean to central only case using the HOD parameters from pG
cen = ksz.Satellites(M_vals, M0=10**(12.92), M1=10**(13.95), M_min=10**(12.89), nM=n_M)
M_mean_cen = cen.mean_halo_mass()
print(f"{M_mean_cen:.2e}")
print(f"{M_mean_sat:.2e}")

#%%

plt.plot(k_arr, pkt_gk1, label=r'$P_{q_\perp,1}$', color='tab:blue')
plt.plot(k_arr, pkt_gk1_1h, label=r'$P_{q_\perp,1}^{\rm (1h)}$', color='tab:blue', linestyle='dashed')
plt.plot(k_arr, pkt_gk1_2h, label=r'$P_{q_\perp,1}^{\rm (2h)}$', color='tab:blue', linestyle='dotted')
plt.plot(k_arr, -pkt_gk2, label=r'$-P_{q_\perp,2}$', color='tab:cyan')
plt.plot(k_arr, pkt_gk, label=r'$P_{q_\perp, \rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=2, loc="best")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkp_gk1, label=r'$P_{q_\parallel,1}$', color='tab:blue')
plt.plot(k_arr, pkp_gk1_1h, label=r'$P_{q_\parallel,1}^{\rm (1h)}$', color='tab:blue', linestyle='dashed')
plt.plot(k_arr, pkp_gk1_2h, label=r'$P_{q_\parallel,1}^{\rm (2h)}$', color='tab:blue', linestyle='dotted')
plt.plot(k_arr, pkp_gk2, label=r'$P_{q_\parallel,2}$', color='tab:cyan')
plt.plot(k_arr, pkp_gk, label=r'$P_{q_\parallel,\rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkt_gk1, label=r'$P_{q_\perp,1}^{\rm (cen \, only)}$', color='tab:blue')
plt.plot(k_arr, pkt_gk1_sat, label=r'$P_{q_\perp,1}^{\rm (cen+sats)}$', color='tab:blue', linestyle='dashed')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=1, loc="best")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse_satellites.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkt_gk1, label=r'$P_{q_\perp,1}$', color='tab:blue')
plt.plot(k_arr, -pkt_gk2, label=r'$-P_{q_\perp,2}$', color='tab:cyan')
plt.plot(k_arr, pkt_tri_4h, label=r'$P_{q_\perp, \rm c}$', color='tab:purple')
plt.plot(k_arr, pkt_gk, label=r'$P_{q_\perp, \rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=2, loc="best")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse_with_cng.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkp_gk1, label=r'$P_{q_\parallel,1}$', color='tab:blue')
plt.plot(k_arr, pkp_gk2, label=r'$P_{q_\parallel,2}$', color='tab:cyan')
plt.plot(k_arr, pkp_bi_1h, label=r'$P_{q_\parallel, B^{\rm (1h)}}$', color='tab:pink')
plt.plot(k_arr, pkp_bi_3h, label=r'$P_{q_\parallel, B^{\rm (3h)}}$', color='tab:purple')
#plt.plot(k_arr, P_mm, label=r'$P_{q_\parallel, B^{\rm (3h)}}$', color='tab:orange')
plt.plot(k_arr, pkt_gk, label=r'$P_{q_\parallel, \rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=1, loc="best")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal_with_bispectrum.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Calculate angular power spectra

#%%

ells = np.geomspace(2, 1e4, 256)

# Galaxy-electron cross-correlation (transverse)
Clt_gk = kSZ.get_Cl(ells, kind="perp", ab="eg")

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
Clt_gk_1h = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="1h")

# Galaxy-kSZ cross-correlation (transverse) for two-halo term only
Clt_gk_2h = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="2h")

# Galaxy-kSZ cross-correlation (transverse) with satellites
Clt_gk_sat = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="sat")

#%%

# Galaxy-kSZ cross-correlation (longitudinal)
Clp_gk = kSZ.get_Cl(ells, kind="par", ab="eg")

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
Clp_gk_1h = kSZ.get_Cl(ells, kind="par", ab="eg", variant="1h")

# Galaxy-kSZ cross-correlation (longitudinal) for two-halo term only
Clp_gk_2h = kSZ.get_Cl(ells, kind="par", ab="eg", variant="2h")

#%%

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:blue", label=r'$D_{\ell, \perp, T}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:blue", label=r'$-D_{\ell, \perp, 2}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -(Clp_gk[0]+Clp_gk[1])), color="tab:red", label=r'$D_{\ell, \parallel, T}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:red", label=r'$D_{\ell, \parallel, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:red", label=r'$D_{\ell, \parallel, 2}$', linestyle='dotted')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="center right", ncol=2)
plt.show()

#%%

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_1h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_2h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:cyan", label=r'$-D_{\ell, \perp, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:red", label=r'$D_{\ell, \perp, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=16, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_transverse.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk_1h[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk_2h[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:cyan", label=r'$D_{\ell, \parallel, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, -(Clp_gk[0]+Clp_gk[1])), color="tab:red", label=r'$D_{\ell, \parallel, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=16, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_longitudinal.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (cen \, only)}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_sat[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (cen+sats)}$', linestyle='dashed')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=16, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_transverse_satellites.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%

pk2d_tri_4h = (pkt_tri_4h / aHf[0]).reshape(128, 1) * aHf_arr.reshape(1, 32)

tg, tk = kSZ._get_tracers(kind='perp')
prefac = ells * (ells+1) / (ells+0.5)**2

pk_tri_arr = pk2d_tri_4h.T
pk_tri = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_tri_arr, is_logp=False)
Clt_tri_4h = prefac * ccl.angular_cl(cosmo, tg, tk, ells, p_of_k_a=pk_tri)

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:cyan", label=r'$-D_{\ell, \perp, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_tri_4h), color="tab:purple", label=r'$D_{\ell, \perp, \rm c}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:red", label=r'$D_{\ell, \perp, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=16, frameon=False, loc="best", ncol=2)
#plt.savefig('kSZ_angular_power_spectra_transverse_with_cng.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%

pk2d_bi_1h = (pkp_bi_1h / aHf[0]).reshape(128, 1) * aHf_arr.reshape(1, 32)
pk2d_bi_3h = (pkp_bi_3h / aHf[0]).reshape(128, 1) * aHf_arr.reshape(1, 32)

tg, tk = kSZ._get_tracers(kind='par')

pk_bi_1h_arr = pk2d_bi_1h.T
pk_bi_1h = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_bi_1h_arr, is_logp=False)
Clp_bi_1h = ccl.angular_cl(cosmo, tg, tk, ells, p_of_k_a=pk_bi_1h)

pk_bi_3h_arr = pk2d_bi_3h.T
pk_bi_3h = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_bi_3h_arr, is_logp=False)
Clp_bi_3h = ccl.angular_cl(cosmo, tg, tk, ells, p_of_k_a=pk_bi_3h)

plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:cyan", label=r'$D_{\ell, \parallel, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_bi_1h), color="tab:pink", label=r'$D_{\ell, \parallel, B^{\rm (1h)}}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_bi_3h), color="tab:purple", label=r'$D_{\ell, \parallel, B^{\rm (3h)}}$')
plt.plot(ells, kSZ.get_Dl(ells, -(Clp_gk[0]+Clp_gk[1])), color="tab:red", label=r'$D_{\ell, \parallel, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=14, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_longitudinal_with_bispectrum.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%
# Calculate covariance

#%%

# Galaxy noise
sigma_v = 300e3 / ccl.physical_constants.CLIGHT
ng_srad = 150 * (180/np.pi)**2 # galaxies per square radian
nl_gg = np.ones_like(ells) * sigma_v**2 / ng_srad

# CMB power spectrum
T_CMB_uK = cosmo['T_CMB'] * 1e6
d = np.loadtxt('data/camb_93159309_scalcls.dat', unpack=True)
ells_cmb = d[0]
D_ells_cmb = d[1]
C_ells_cmb = 2 * np.pi * D_ells_cmb / (ells_cmb * (ells_cmb+1) * T_CMB_uK**2)
nl_TT_cmb = interp1d(ells_cmb, C_ells_cmb, bounds_error=False, fill_value=0)(ells)

# Secondary anisotropies
d = pickle.load(open("data/P-ACT_theory_cells.pkl", "rb"))
ells_act = d["ell"]
D_ells_act = d["tt", "dr6_pa5_f090", "dr6_pa5_f090"] 
C_ells_act = 2 * np.pi * D_ells_act / (ells_act * (ells_act+1) * T_CMB_uK**2)
nl_TT_act = interp1d(ells_act, C_ells_act, bounds_error=False, fill_value=0)(ells)

# SO CMB noise
d = np.loadtxt("data/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt", unpack=True)
ells_n = d[0]
nl_so = d[1]
nl_so /= T_CMB_uK**2
#nl_TT_cmb += interp1d(ells_n, nl_so, bounds_error=False, fill_value=(nl_so[0], nl_so[-1]))(ells)
#nl_TT_act += interp1d(ells_n, nl_so, bounds_error=False, fill_value=(nl_so[0], nl_so[-1]))(ells)

# S4 CMB noise
lknee = 2154
aknee = -3.5 # Atmospheric noise
DT = 2.0 # Noise rms: 2 uK-arcmin
fwhm = 1.4 # Beam FWHM
beam = np.exp(-0.5*ells_n*(ells_n+1)*(np.radians(fwhm/60)/2.355)**2)
Nwhite = DT**2*(np.pi/180/60)**2 # White amplitude in uK^2 srad
nl_s4 = Nwhite * (1 + (ells_n/lknee)**aknee)/beam**2
nl_s4 /= T_CMB_uK**2
nl_s4 = 1/(1/nl_s4+1/nl_so)
#nl_TT_cmb += interp1d(ells_n, nl_s4, bounds_error=False, fill_value=(nl_s4[0], nl_s4[-1]))(ells)
#nl_TT_act += interp1d(ells_n, nl_s4, bounds_error=False, fill_value=(nl_s4[0], nl_s4[-1]))(ells)

#%%

# Transverse auto-correlations
Clt_gg = kSZ.get_Cl(ells, kind="perp", ab="gg")
Clt_kk = kSZ.get_Cl(ells, kind="perp", ab="ee")

Clt_gk_T = Clt_gk[0] + Clt_gk[1]
Clt_gg_T = Clt_gg[0] + Clt_gg[1]
Clt_kk_T = Clt_kk[0] + Clt_kk[1]

f_sky = 0.5

#%%
# Knox formula to calculate variance
var = ((Clt_gg_T + nl_gg) * (-Clt_kk_T*0 + nl_TT_cmb) + (Clt_gk[0]+Clt_gk[1])**2) / ((2 * ells + 1) * f_sky)

plt.plot(ells, (Clt_gk[0]+Clt_gk[1]), color='tab:blue')
plt.errorbar(ells, (Clt_gk[0]+Clt_gk[1]), yerr=np.sqrt(var), color='tab:blue', alpha=0.5)
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell, \perp}^{\pi T}$', fontsize=24)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.show()

#%%

plt.plot(ells, nl_TT_cmb)
plt.plot(ells, nl_TT_act)
plt.loglog()
plt.show()

#%%
# Calculate signal-to-noise

#%%

def S_to_N(Cl1, Cl2, var):
    delta_ell = np.gradient(ells)
    s2n = np.sqrt(np.sum(delta_ell * (Cl1 - Cl2)**2 / var))
    return s2n
  
# kSZ-only
delta_ell = np.gradient(ells)
sn_ksz = np.sqrt(np.sum(delta_ell * Clt_gk_T**2 / var))
print('S/N kSZ =', sn_ksz)

# Sub-dominant contribution
sn_sd = S_to_N(Clt_gk[0]*0, Clt_gk[1], var)
print('S/N Perp_1 vs Perp_2 =', sn_sd)

# Trispectrum contribution
sn_tri = S_to_N(Clt_gk[0]*0, Clt_tri_4h, var)
print('S/N Total vs Trispectrum Term =', sn_tri)

# Longitudinal mode
sn_par = S_to_N((Clt_gk[0]+Clt_gk[1])*0, (Clp_gk[0]+Clp_gk[1]), var)
print('S/N Perp vs Par =', sn_par)

# One-halo term only
sn_1h = S_to_N(Clt_gk[0], Clt_gk_1h[0], var)
print('S/N 1h+2h vs 1h =', sn_1h)

# Satelite galaxies
s2n_sat = S_to_N(Clt_gk[0], Clt_gk_sat[0], var)
print('S/N cen+sat vs sat =', s2n_sat)

#%%
# Higher-order contributions

#%%

from scipy.interpolate import RectBivariateSpline

# Define grids for M and k
log10M_grid = np.linspace(11, 15, 128)
M_grid = 10**log10M_grid
k_grid = np.logspace(-3, 2, 128)
log10k_grid = np.log10(k_grid)

# Arrays for Fourier transforms
u_pM_grid = np.zeros((len(k_grid), len(M_grid)))
u_pG_grid = np.zeros((len(k_grid), len(M_grid)))
u_pE_grid = np.zeros((len(k_grid), len(M_grid)))

# Fill the grids
for i, k_val in enumerate(k_grid):
    for j, M_val in enumerate(M_grid):
        u_pM_grid[i, j] = pM.fourier(cosmo, k_val, M_val, a) / M_val
        u_pG_grid[i, j] = pG.fourier(cosmo, k_val, M_val, a) / pG.get_normalization(cosmo, a, hmc=hmc)
        u_pE_grid[i, j] = pGas.fourier(cosmo, k_val, M_val, a)

pM_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pM_grid)
pG_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pG_grid)
pE_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pE_grid)

k_dense = np.logspace(-3, 1, 1000)
P_dense = ccl.linear_matter_power(cosmo, k_dense, a)
P_L_interp = interp1d(k_dense, P_dense, bounds_error=False, fill_value=0.0)

#%%

test_k = 1e-2

ho = ksz.HigherOrder(cosmo, k_arr, a, M_vals, n_M, b_M, pM_interp, pG_interp, pE_interp, P_L_interp) 

Pt_tri_4h = ho.compute_P(test_k, spectra_type="trispectrum", kind="perp", term="4h") 

Pp_tri_1h = ho.compute_P(test_k, spectra_type="trispectrum", kind="par", term="1h") 
Pp_tri_4h = ho.compute_P(test_k, spectra_type="trispectrum", kind="par", term="4h") 

Pp_bi_1h_g = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="1h", density_kind='galaxy') 
Pp_bi_3h_g = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="3h", density_kind='galaxy')

Pp_bi_1h_e = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="1h", density_kind='electron') 
Pp_bi_3h_e = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="3h", density_kind='electron')
