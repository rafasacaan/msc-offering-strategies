# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:35:58 2016

@author: nimaz
"""

#%% Import Packages
import os
import pandas as pd
import numpy as np
import numba as nb
import math
import scipy
from scipy import interpolate
from scipy import integrate
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
import random
from collections import Counter


def generateForecast(which_day,num_scenarios,plot=True):
    lab_size    = 20
    ax_lab_size = 16
    leg_size    = 18

    #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{eurosym}',
                               r'\usepackage{amsmath,amsfonts,amsthm}' ])

    random.seed(1234)

    os.path.dirname(os.path.realpath('__file__'))
    pkl_file = open('data/wind_forecast_data.pkl', 'rb')
    Data_dict = pickle.load(pkl_file)
    #%% Ciao
    Hourly_dict=Data_dict['Hourly']
    Hour_idx = Hourly_dict['Hour_idx']
    Day_idx = Hourly_dict['Day_idx']
    Operational_fore_dict = Hourly_dict['Fore_dict']
    W_max_real=Data_dict['W_max']
    #%% Function definition
    def phi(x):
        'Cumulative distribution function for the standard normal distribution'
        return (1.0 + scipy.special.erf(x / 2**0.5)) / 2.0

    def probit(x):
        'Cumulative distribution function for the standard normal distribution'
        return 2**0.5*scipy.special.erfinv(2*x-1)

    def supply_curve_util(x,wind,beta,gamma):
        if x <= wind:
            y = 0
        else:
            y=beta*(x-wind)+gamma*(x-wind)**2
        return y

    def supply_curve(x_vect,wind,beta,gamma):
        y_vect = np.array([supply_curve_util(x,wind,beta,gamma) for x in x_vect])
        return y_vect    

    def fun_muvar(x,v=1,x0=0,x1=1):
        x_n = (x-x0)/(x1-x0)
        y = 4*(x_n**v)*(1-x_n**v)
        return y

    def func_exp(x, par_exp):
        return par_exp[0] * np.log(par_exp[1]*x) + par_exp[2]

    def lambda_spot_eval(alpha,beta,gamma,delta,wind):        
        a = gamma
        b = beta - delta - 2*gamma*wind
        c = gamma*wind**2 - wind*beta - alpha
        Q = (- b + (b**2 - 4*a*c)**0.5)/(2*a)
        lambda_sp = alpha + delta*Q
        if Q > wind:
            lambda_spot = lambda_sp
        else:
            lambda_spot = 0
        return lambda_spot

    def lambda_bm_eval(beta,gamma,Q,R):
        lambda_bm = [supply_curve_util(Q[i],R[i],beta,gamma[i]) for i in range(len(Q))]
        return lambda_bm

    def lambda_bm_eval_util(a,b,c,x0,x):
        if x > x0:
            y = a*x**2  + b*x  + c
        else:
            y = a*x0**2 + b*x0 + c
        return y

    def lambda_bm_eval2(beta,gamma,ratio_gamma,lambda0,Q_spot,lambda_spot,DeltaR):
        gamma_1 = ratio_gamma*gamma
        a,b,c = 1/(4*gamma_1),Q_spot,lambda0 - lambda_spot + gamma_1*Q_spot**2
        a_new,b_new = gamma_1,(- b + (b**2 - 4*a*c)**0.5)/(2*a)
        c_new = lambda_spot - a_new*Q_spot**2 - b_new*Q_spot
        Q0 = -b_new/(2*a_new)   
        lambda_bm = [lambda_bm_eval_util(a_new[i],b_new[i],c_new[i],Q0[i],Q_spot[i]+DeltaR[i]) for i in range(len(DeltaR))]
        return lambda_bm

    def lambda_bm_eval2b(beta,gamma,ratio_gamma,lambda0,Q_spot,lambda_spot,DeltaR):
        gamma_1 = ratio_gamma*gamma
        a,b,c = 1/(4*gamma_1),Q_spot,lambda0 - lambda_spot + gamma_1*Q_spot**2
        a_new,b_new = gamma_1,(- b + (b**2 - 4*a*c)**0.5)/(2*a)
        c_new = lambda_spot - a_new*Q_spot**2 - b_new*Q_spot
        Q0 = -b_new/(2*a_new)   
        lambda_bm = [lambda_bm_eval_util(a_new[0],b_new[0],c_new[0],Q0[0],Q_spot[0]+DeltaR[i]) for i in range(len(DeltaR))]
        return lambda_bm

    def F_cdf_interp(x,q5,q95,spline_int,W_max,rho):
        if x==0:
            out = 0
        if 0<x<q5:
            out = rho*math.exp(q5/x*math.log(0.05/rho))
        if q5<=x<=q95:
            out = interpolate.splev(x, spline_int, der=0)
        if q95<x<W_max:
            out = 1 - rho*math.exp((W_max-q95)/(W_max-x)*math.log(0.05/rho))
        if x==W_max:
            out = 1
        return out

    def F_inv_interp(y,q5,q95,spline_int,W_max,rho):
        if y==0:
            out = 0
        if 0<y<0.05:
            out = q5*math.log(0.05/rho)/math.log(y/rho)
        if 0.05<=y<=0.95:
            out = interpolate.splev(y, spline_int, der=0)
        if 0.95<y<1:
            out = W_max - (W_max-q95)*math.log(0.05/rho)/math.log((1-y)/rho)
        if y==1:
            out = W_max
        return out

    def F_cdf(x_vect,quant_vect,W_max,rho=50):
        tck = interpolate.splrep(quant_vect,np.linspace(0.05,0.95,19))
        y_cdf = np.array([F_cdf_interp(x,quant_vect[0],quant_vect[18],tck,W_max,rho) for x in x_vect])
        return y_cdf

    def F_inv(y_vect,quant_vect,W_max,rho=50):
        tck = interpolate.splrep(np.linspace(0.05,0.95,19),quant_vect)
        x_cdf = np.array([F_inv_interp(y,quant_vect[0],quant_vect[18],tck,W_max,rho) for y in list(y_vect)])
        return x_cdf
    #%% scenario reduction (numba functions)
    @nb.jit('i4(f4[:], f4[:], f4[:])',nopython=True)
    def step1_1D(P_in,P_out,dist):
        Nout = len(P_out)
        Nin = len(P_in)
        Nd = len(dist)
        for i in range(Nout):
            Sigma = 0
            for p in range(Nout):
                if p != i:
                    d = abs(P_out[i] - P_out[p])
                    for j in range(Nin):
                        dx = abs(P_in[j] - P_out[p])
                        if dx < d:
                            d = dx
                    Sigma = Sigma + d
            dist[i] = Sigma
        bm = dist[0]
        j = 0
        for i in range(Nd):
            if dist[i] < bm:
                bm = dist[i]
                j = i
        return j

    @nb.jit('f4[:](f4[:], f4[:], f4[:])',nopython=True)
    def step2_1D(P_in,P_out,prob_list):        
        Nout = len(P_out)
        Nin = len(P_in)
        for i in range(Nout):
            i_out = P_out[i]
            j_idx = 0
            bm = abs(P_in[0]-i_out)
            for j in range(Nin):
                dx = abs(P_in[j]-i_out)
                if dx < bm:
                    bm = dx
                    j_idx = j
            prob_list[i] = j_idx
        return prob_list

    def oneD_reduction(P_sample,Ng):
        P_in = np.array([],dtype='f4')
        P_out = np.array(P_sample,dtype='f4')
        for gg in range(Ng):
            idx_min = step1_1D(P_in,P_out,np.zeros(len(P_out),dtype='f4'))
            P_in = np.append(P_in,P_out[idx_min])
            P_out = np.append(P_out[:idx_min],P_out[idx_min+1:])
        prob_list = step2_1D(P_in,P_out,np.zeros(len(P_out),dtype='f4'))
        con = Counter(prob_list)
        pi_list = np.array([(con[i]+1)/len(P_sample) for i in range(len(P_in))])
        return P_in,pi_list
    #%%
    #rho,mux,muy,sigmax,sigmay = 0.8669,0,0,1,1

    def fy(x,y,mux=0,muy=0,sigmax=1,sigmay=1,rho=0.8669):
        f = 1/(2*math.pi*sigmax*sigmay*(1-rho**2)**.5)*math.exp(-1/(2*(1-rho**2))*((x-mux)**2/sigmax**2+(y-muy)**2/sigmay**2-2*rho*(x-mux)*(y-muy)/(sigmax*sigmay)))*1/(norm.pdf(x))
        return f
    #%% Function definition
    pkl_file = open('data/DAM_params.pkl', 'rb')
    DAM_params = pickle.load(pkl_file)
    W_max = DAM_params['others']['W_max']
    #%% Function definition
    K_wind = 24
    v_wind = 7
    NΩ = num_scenarios   # number of scenarios MODIFY  
    ratio_gamma  = DAM_params['others']['ratio_gamma']
    lambda0 = DAM_params['others']['lambda0']
    horizon = 12     # look-ahead time MODIFY
    H_idx = ['h{0}'.format(h) for h in range(horizon+1)]
    J_idx = ['j{0}'.format(i+1) for i in range(NΩ)]
    #%%
    day = which_day #MODIFY

    dam_params=DAM_params['day_'+str(day)]
    pkl_file = open('data/Scen_DAM_day'+str(day)+'.pkl', 'rb')
    dam_scen = pickle.load(pkl_file)
    Day_df = pd.DataFrame({'beta':[dam_params['beta']]*24,'gamma':dam_params['gamma_real'],'ratio_gamma':[DAM_params['others']['ratio_gamma']]*24,'lambda0':[DAM_params['others']['lambda0']]*24,'Q_dam_real':[dam_scen['Q_dam_d1']['t'+str(t+1)]for t in range(24)],'lambda_spot_real':[dam_scen['lambda_dam_real']['t'+str(t+13)]for t in range(24)],'exp_wind':[dam_scen['Wind_dam_d1']['t'+str(t+1)]for t in range(24)]},index=range(24))
    dam_params=DAM_params['day_'+str(day+1)]
    pkl_file = open('data/Scen_DAM_day'+str(day+1)+'.pkl', 'rb')
    dam_scen = pickle.load(pkl_file)
    Day_df1 = pd.DataFrame({'beta':[dam_params['beta']]*24,'gamma':dam_params['gamma_real'],'ratio_gamma':[DAM_params['others']['ratio_gamma']]*24,'lambda0':[DAM_params['others']['lambda0']]*24,'Q_dam_real':[dam_scen['Q_dam_d1']['t'+str(t+1)]for t in range(24)],'lambda_spot_real':[dam_scen['lambda_dam_real']['t'+str(t+13)]for t in range(24)],'exp_wind':[dam_scen['Wind_dam_d1']['t'+str(t+1)]for t in range(24)]},index=range(24,48))
    Day_df=pd.concat([Day_df,Day_df1])

    #%%
    Bm_dict = {}

    for hour in range(24):
        Quant_array = Operational_fore_dict[Day_idx[day],Hour_idx[hour]]
        MData=Day_df.iloc[hour:(hour+13)]
        Quant_array.index=['h{0}'.format(h+1) for h in range(12)]
        Quant_array = Quant_array.loc[['h{0}'.format(h+1) for h in range(horizon)]]
        yunif = np.linspace(1/(2*NΩ),(2*NΩ-1)/(2*NΩ),NΩ)
        Wind_Scen=pd.DataFrame()
        for h in range(horizon):
            quant_vect = Quant_array.loc['h'+str(h+1),'q5':]/W_max_real
            Wind_Scen['h'+str(h+1)]=F_inv(yunif,quant_vect,1)
        Scen_mtx = np.array(Wind_Scen[['h{0}'.format(h+1) for h in range(horizon)]])
        wind_real = Quant_array['pers'][0]/W_max_real
        λ_DA0 = list(MData['lambda_spot_real'])[0]
        λ_BA0 = lambda_bm_eval2(np.array([list(MData['beta'])[0]]),np.array([list(MData['gamma'])[0]]),np.array([list(MData['ratio_gamma'])[0]]),np.array([list(MData['lambda0'])[0]]),np.array([list(MData['Q_dam_real'])[0]]),np.array([list(MData['lambda_spot_real'])[0]]),np.array([-wind_real*W_max+np.array(list(MData['exp_wind'])[0])]))[0]
        lambda_BA_dict,lambda_BA_ay = {},np.zeros((horizon+1,NΩ))
        for j in range(NΩ):
            lambda_BA_dict[(H_idx[0],J_idx[j])] = λ_BA0
            lambda_BA_ay[0,j] = λ_BA0
        for h in range(horizon):
            λ_BA = lambda_bm_eval2b(np.array([list(MData['beta'])[h+1]]),np.array([list(MData['gamma'])[h+1]]),np.array([list(MData['ratio_gamma'])[h+1]]),np.array([list(MData['lambda0'])[h+1]]),np.array([list(MData['Q_dam_real'])[h+1]]),np.array([list(MData['lambda_spot_real'])[h+1]]),np.array(-Scen_mtx[:,h]*W_max+np.array(list(MData['exp_wind'])[h+1])))
            for j in range(NΩ):
                lambda_BA_dict[(H_idx[h+1],J_idx[j])] = λ_BA[j]
                lambda_BA_ay[h+1,j] = λ_BA[j]        
        Bm_dict_h={}
        Bm_dict_h['lambda_bm_scend'] = lambda_BA_dict
        Bm_dict_h['lambda_bm_scena'] = lambda_BA_ay
        Bm_dict_h['lambda_da_real']  = λ_DA0
        Bm_dict['hour'+str(hour+1)]=Bm_dict_h

    #%%        
    xnorm = probit(yunif)
    xlims = probit(np.linspace(0.001,0.999,NΩ+1))
    Prob_ay = np.zeros((NΩ,NΩ))
    for j1 in range(NΩ):
        for j2 in range(NΩ):
            Prob_ay[j1,j2] = integrate.quad(lambda x: fy(xnorm[j1],x),xlims[j2],xlims[j2+1])[0]


    #%%
    if plot:
        col_DA = '#0066ff'
        col_BA = '#ff0000'
        col_scen = '#8c8c8c'

        t_DA = [t for t in range(1,25)]
        λ_DA = [Bm_dict['hour'+str(hour)]['lambda_da_real'] for hour in range(1,25)]

        t_BA = [t for t in range(1,25)]
        λ_BA = [Bm_dict['hour'+str(hour)]['lambda_bm_scena'][0,0] for hour in range(1,25)]

        hora = 20 # MODIFY to plot tree from hour 'hora'

        λ_scen = Bm_dict['hour'+str(hora)]['lambda_bm_scena'] #mod
        t_scen = np.zeros(λ_scen.shape)
        for h in range(λ_scen.shape[0]):
            t_scen[h,:] = np.ones(λ_scen.shape[1])*(h+1+hora-1)#mod

        fig, ax = plt.subplots(1,figsize=(8,5))
        ax.plot(t_DA,λ_DA,ls='dotted',lw=2.0,color=col_DA,label=r'$\lambda^{\mathrm{DA}}$ real')
        ax.plot(t_BA,λ_BA,ls='-',lw=2.0,color=col_BA,label=r'$\lambda^{\mathrm{BA}}$ real')

        for j in range(NΩ):
            ax.plot([t_scen[0,0],t_scen[1,j]],[λ_scen[0,0],λ_scen[1,j]],ls="-",lw=.3,color="k")

        for h in range(1,horizon):#(1,horizon)
            for j1 in range(NΩ):
                for j2 in range(NΩ):
                    ax.plot([t_scen[h,j1],t_scen[h+1,j2]],[λ_scen[h,j1],λ_scen[h+1,j2]],ls="-",lw=.3,color="k")

        ax.plot(t_scen[0,0],λ_scen[0,0],ls='',marker="o",markersize=7,color=col_BA)
        ax.plot(t_scen[1:,:],λ_scen[1:,:],ls='',marker="o",markersize=5,color=col_scen)



        ax.set_xlim(.7,24.3)
        ax.set_xticks([4,8,12,16,20,24])
        ax.set_xticklabels([4,8,12,16,20,24],size=ax_lab_size)
        ax.set_xlabel(r"time (h)",size=lab_size)
        ax.set_ylim(0,90)
        ax.set_yticks([0,10,20,30,40,50,60,70,80,90])
        ax.set_yticklabels([0,10,20,30,40,50,60,70,80,90],size=ax_lab_size)
        ax.set_ylabel(r"price (\pounds/MWh)",size=lab_size)
        ax.legend(fontsize='x-large')

        fname = 'hour1'
        plt.savefig("./images/"+ fname +".pdf", format="pdf")
    
    return(Bm_dict, Prob_ay)


