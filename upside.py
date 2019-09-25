#from scen_generation_new import Bm_dict, Prob_ay
import numpy as np
import time

import pyomo.environ
from pyomo.core import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
#%matplotlib notebook

def runDP(hour, params, data, data2, probMat):
    ''' Run Dynamic Programming: 
    Run backwards pass using DP to evaluate a value function. 
    
    Inputs:
    hour: current hour (hour = 1,2,3,...) - [int]
    params: parameters of simulation and energy units - [dict]
    data: price scenarios and forecast - [dict] ('Bm_dict' from 'scen_generation_new.py')
    probMat: matrix with node transition probabilities
    data1: day 1
    data2: day 2
    
    Output:
    FF: Value function F(t,s) for time 't' and price scenario 's' - [dict]
    '''

    # Record start time
    startTime = time.time()

    # Hours: once inside 'Bm_dict[hourX]', a forecast is given from the current hour and 
    # on (h0,h1,h2,...,h12). 
    h = hour                                                      # load data for current hour 'h'
    tstartDP = 4                                                  # obtain value functions for this hour 
    Ns = len(probMat)                                             # number of scenarios per stage
    tendDP = int(len(data['hour{}'.format(h)]['lambda_bm_scend'])/Ns) - 1 # last time/stage
    S = list(range(1,Ns+1))                                       # num. of price scenarios in each time stage

    # Node tree
    Node = {}
    for t in range(tstartDP,tendDP+1):
        for s in S:        
            Node.update({
                (t,s): {
                    'price': data['hour{0}'.format(h)]['lambda_bm_scend']['h{0}'.format(t),'j{0}'.format(s)], 
                }
            })

            
    # Get qDA
    # [day1(24 hrs), day2 (12 hrs)]
    # [0, h1, h2, ...., h24, h25, ..,h36]
    qDA = [data['hour{}'.format(h)]['lambda_da_real'] for h in range(1,25)]
    qDA.insert(0,0)
    qDA_day2 = [data2['hour{}'.format(h)]['lambda_da_real'] for h in range(1,13)]
    qDA.extend(qDA_day2)
    
    
    # Decision variables:
    # pup : quantity charged to battery during time t 
    # pdw : quantity discharged by battery during time t 
    # r   : ramp production with respect to last period by the diesel
    Pup = np.array(range(0,params['PUP']+1,params['RES']))  
    Pdw = np.array(range(0,params['PDW']+1,params['RES']))  
    R   = np.array(range(-params['DRAMP'],params['DRAMP']+1,params['RES'])) 

    # State variables:
    # d   : diesel generating status at start of period t
    # l   : level of storage at start of period t
    D = np.array(range(0,params['DMAX']+1,params['RES']))
    L = np.array(range(0,params['LMAX']+1,params['RES']))


    # Value function: 
    # FF(t=time, s=price_scenario, d=thermal unit state, l= storage level)
    # t : time period
    # s : price scenario
    # d : diesel unit level at start of period t
    # l : storage level at start of period t
    FF = { (tendDP+1,s,d,l): 0 for s in S for d in D for l in L }
 
       
    # Start backwards evaluation (DP)
    for t in range(tendDP,tstartDP-1,-1):
        for s in S:
            BMprice = Node[t,s]['price']
            #DAprice = data['hour{}'.format(h+t)]['lambda_da_real']
            DAprice = qDA[h+t]
            upReg = BMprice >= DAprice
            for d in D:
                for l in L:
                    decisions_values = []
                    decisions = []
                    for pup in Pup:
                        for pdw in Pdw:
                            for r in R:          
                                
                                if upReg:

                                    if (d + r) <= params['DMAX'] and (d + r) >= params['DMIN'] and \
                                       (l - pdw + pup) <= params['LMAX'] and (l - pdw + pup) >= params['LMIN'] and \
                                       (d + r - pup + pdw) >= params['QDA']:

                                        q = d + r - pup + pdw
                                        lNext = l - pdw + pup
                                        dNext = d + r

                                        # Original working VF
                                        #val = Node[t,s]['price'] * q - params['MC'] * (dNext) + \
                                        #      sum([probMat[s-1][sNext-1] * FF[t+1,sNext,dNext,lNext] for sNext in S])

                                        val=DAprice*params['QDA'] + BMprice*(q-params['QDA']) - params['MC']*(dNext) + \
                                            sum([probMat[s-1][sNext-1] * FF[t+1,sNext,dNext,lNext] for sNext in S])

                                        decisions_values.append(val)
                                        decisions.append((pup,pdw,r))
                                    else:
                                        decisions_values.append(params['MDP'])
                                        decisions.append((pup,pdw,r))
                                
                                else:
                                    
                                    if (d + r) <= params['DMAX'] and (d + r) >= params['DMIN'] and \
                                       (l - pdw + pup) <= params['LMAX'] and (l - pdw + pup) >= params['LMIN'] and \
                                       (d + r - pup + pdw) <= params['QDA']:

                                        q = d + r - pup + pdw
                                        lNext = l - pdw + pup
                                        dNext = d + r

                                        # Original working VF
                                        #val = Node[t,s]['price'] * q - params['MC'] * (dNext) + \
                                        #      sum([probMat[s-1][sNext-1] * FF[t+1,sNext,dNext,lNext] for sNext in S])

                                        val=DAprice*params['QDA'] + BMprice*(q-params['QDA']) - params['MC']*(dNext) + \
                                            sum([probMat[s-1][sNext-1] * FF[t+1,sNext,dNext,lNext] for sNext in S])

                                        decisions_values.append(val)
                                        decisions.append((pup,pdw,r))
                                    else:
                                        decisions_values.append(params['MDP'])
                                        decisions.append((pup,pdw,r))

                    FF[t,s,d,l] = np.amax(decisions_values) 
    
    # Print DP time
    DPTime = time.time()
    print(" ")
    print("Running time (DP): {0:.3f} seconds.".format(DPTime - startTime))
    
    
    
    # Identify and remove datapoints that return infeasible
    # TESTING
    delKeys=[]
    for k,v in FF.items():
        if v < -10**4:
            delKeys.append(k)

    for key in delKeys:
        del FF[key]
    
    
    # Least Squares
    # Calculate a continuous approximation for the value function 'FF'
    # Build list with a matrix (cols: x,y,z) per price scenario
    Data = [0]
    for s in S:
        dataX = []
        dataY = []
        dataZ = []

        for d in D:
            for l in L:
                if (tstartDP,s,d,l) in FF.keys(): #TESTING
                    dataX.append(d)
                    dataY.append(l)
                    dataZ.append(FF[tstartDP,s,d,l])

        dataX, dataY, dataZ = np.array(dataX), np.array(dataY), np.array(dataZ)
        Data.append(np.c_[dataX,dataY,dataZ])

    # Best fit quadratic function (features: 1,x,y,xy,x2,y2)
    CC = [0]  # list with set of coefficients

    for s in S:
        A = np.c_[np.ones(Data[s].shape[0]), Data[s][:,0], Data[s][:,1], np.multiply(Data[s][:,0],Data[s][:,1]),
                  Data[s][:,0]**2, Data[s][:,1]**2 ]
        C, residuals, _, _ = np.linalg.lstsq(A, Data[s][:,2], rcond=None)
        CC.append(C)
        
    # Print Least Squares time
    LSTime = time.time() - DPTime
    print("Running time (LS): {0:.3f} seconds.".format(LSTime))
    
    return(CC,FF)

def plotVF(s, CC, FF, params):
    '''Plot Value Function and approximated plane
    by least squares for a given price scenario (s).
    
    Inputs:
    s = choose price scenario index from value function (s=1,2,3..) - [int]
    CC = least squares coeff. - [list] len=6
    '''
    
    # State variables:
    # d   : diesel generating status at start of period t
    # l   : level of storage at start of period t
    D = np.array(range(0,params['DMAX']+1,params['RES']))
    L = np.array(range(0,params['LMAX']+1,params['RES']))

    dataX = []
    dataY = []
    dataZ = []

    for d in D:
        for l in L:
            if (4,s,d,l) in FF.keys(): #TESTING
                dataX.append(d)
                dataY.append(l)
                dataZ.append(FF[4,s,d,l])

    dataX, dataY, dataZ = np.array(dataX), np.array(dataY), np.array(dataZ)

        
    # Value function (continuous)
    def VF(s,x,y):
        return(CC[s][0] + CC[s][1]*x + CC[s][2]*y + CC[s][3]*x*y + CC[s][4]*x**2 + CC[s][5]*y**2)    
        
    # Print curve for scenario s=1
    # Evaluate it on a grid
    nx, ny = 30, 30
    xx, yy = np.meshgrid(np.linspace(dataX.min(), dataX.max(), nx),
                         np.linspace(dataY.min(), dataY.max(), ny))
    xxr = xx.ravel()
    yyr = yy.ravel()
    AA = np.c_[np.ones(len(xxr)), xxr, yyr, np.multiply(xxr,yyr), xxr**2, yyr**2]

    zz = np.reshape(np.dot(AA, CC[s]), xx.shape)

    fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(zz, cmap=cm.winter, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    
    ax.scatter3D(dataX,dataY,dataZ,c='blue', alpha=.5)
    fg.canvas.draw()

    ax.set_xlabel("thermal unit state")
    ax.set_ylabel("storage level")
    ax.set_zlabel("value")
    plt.show()

def buildOfferCurves():
    ''' Build offer curves for time periods t=0,1,2.
    Returns a [dict].
    '''
    # Offer curves: (price, quantity) pairs.
    offer_curves = {
        0: {'up': {k: 0 for k in range(20,61,10)}, 'dw': {k: 0 for k in range(20,61,10)}},
        1: {'up': {k: 0 for k in range(20,61,10)}, 'dw': {k: 0 for k in range(20,61,10)}},
        2: {'up': {k: 0 for k in range(20,61,10)}, 'dw': {k: 0 for k in range(20,61,10)}},
    }

    # Fill offer curves with initial values
    for t in range(0,3):
        for k,v in offer_curves[t]['up'].items():
            if k>=60:
                offer_curves[t]['up'][k] = 0
        for k,v in offer_curves[t]['dw'].items():
            if k<=20:
                offer_curves[t]['dw'][k] = 0
    
    return(offer_curves)

def runThreeStage(hour, params, data, probMat, offerCurves, L0, D0, CC, useVF=False, log=False):
    ''' Run Three-stage stochastic model.
    Inputs:
    hour: current hour
    data: price scenarios (Bm_dict) - [dict]
    probMat: node transition probabilities (Prob_ay) - [np.array]
    
    Output:
    Offer curve for time 'hour+3'
    
    '''

    ###########################
    # Initial Setup          #
    ##########################
    
    # Scenarios Scheme
    # One-stage decision: go from node (0,0) to nodes in stage 1,2 and 3.
    # Nodes are identified as node[t,n], time stage t and id n.

    # Parameters
    h = hour
    T = 3                                   # number of time/stages [0-3]
    Ns = len(probMat)                       # number of price scenarios per stage
    N = sum([Ns**i for i in range(1,T+1)])  # number of nodes (w/o counting node (0,0))
    #S = Ns ** T                            # number of scenario paths

    # Node tree 
    node = {}

    # t=0
    node.update({
        (0,0): {
            'price': data['hour{0}'.format(h)]['lambda_bm_scend'][('h0','j1')],
            'prob': 1,
            'ancestor': None,
        }
    })

    # t=1,2,3
    for t in range(1,T+1):
        for n in range(1,Ns**t + 1):
            j = n%Ns if n%Ns != 0 else Ns    # scenario mapped to multiples of Ns
            i = (n-1)//Ns + 1 if t!=1 else 0 # ancestor
            i_map = i%Ns if i%Ns != 0 else Ns

            node.update({
                (t,n): {
                    'price': data['hour{0}'.format(h)]['lambda_bm_scend'][('h{0}'.format(t),'j{0}'.format(j))], 
                    'tprob' : probMat[i_map-1][j-1] if t!=1 else 1.0/Ns , # transition prob. from last state
                    'ancestor': (t-1,i),
                    'price_level': j}
            })

    # Set node's absolute probabilities
    for t in range(1,T+1):
        for n in range(1,Ns**t+1):
            if t == 1:
                node[t,n]['prob'] = node[t,n]['tprob']
            else:
                node[t,n]['prob'] = node[t,n]['tprob'] * node[node[t,n]['ancestor']]['prob']


    # Day-ahead
    price_DA = {
        0: data['hour{0}'.format(h  )]['lambda_da_real'],
        1: data['hour{0}'.format(h+1)]['lambda_da_real'],
        2: data['hour{0}'.format(h+2)]['lambda_da_real'],
        3: data['hour{0}'.format(h+3)]['lambda_da_real'],
    }

    q_DA = {
        0: params['QDA'], # generate now, including regulation
        1: params['QDA'],
        2: params['QDA'],
        3: params['QDA'],
    }
 

    # Local functions
    def M_up(n, n_, t):
        if node[t,n]['price'] >= node[t,n_]['price'] and node[t,n]['price'] > price_DA[t]:
            return 1
        else:
            return 0

    def M_dw(n, n_, t):
        if node[t,n]['price'] <= node[t,n_]['price'] and node[t,n]['price'] < price_DA[t]:
            return 1
        else:
            return 0

    def anc(t,n):
        return node[t,n]['ancestor'][1]

    def nodes_in(t):
        # return id of nodes in stage t
        ret = []
        for i,tupl in enumerate(node.keys()):
            if tupl[0] == t:
                ret.append(tupl[1])
        return ret

    def offer_curve_as_fcn(t, reg, off_curves):
        # outputs a piecewise function
        # the output of the function is the cleared 'quantity' for a 'price' input
        p,q=0,0
        if reg == 'up':
            sorted_offer_curve = {k: off_curves[t][reg][k] for k in sorted(off_curves[t][reg].keys())}
            p = list(sorted_offer_curve.keys())
            q = list(sorted_offer_curve.values())
        elif reg == 'dw':
            sorted_offer_curve = {k: off_curves[t][reg][k] for k in sorted(off_curves[t][reg].keys())}
            p = list(sorted_offer_curve.keys())
            p.insert(0,0)
            q = list(sorted_offer_curve.values())
            q.append(0)
        def fcn(x):
            return float(np.piecewise(x, [x >= num for num in p], q))
        return fcn
    
    
    ###########################
    # Build Three-stage model #
    ###########################
    
    # Pyomo Model
    model = ConcreteModel()
    model.Stages = range(0,T+1)
    model.NodesId = range(0,N+1+1) # include node (0,0)
    model.Nodes = node.keys()

    # Variables
    model.o_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.o_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.q_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.q_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.rho_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000000000))
    model.rho_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000000000))

    model.d = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.l = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.c = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000000000000))
    model.p_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.p_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.u = Var(model.Nodes,within=Binary)
    model.y = Var(model.Nodes,within=Binary)
    model.z = Var(model.Nodes,within=Binary)
    
    model.vi_plus  = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000)) # virtual generator / penalty
    model.vi_minus = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000))
    
    # Value function (continuous)
    def VF(s,x,y):
        return(CC[s][0] + CC[s][1]*x + CC[s][2]*y + CC[s][3]*x*y + CC[s][4]*x**2 + CC[s][5]*y**2)
        
        
    # Objective (nodes)
    if useVF:
        model.obj = Objective(
            expr= sum( 
                    node[tupl]['prob'] * (model.rho_up[tupl] - model.rho_dw[tupl] - model.c[tupl]) 
                    for tupl in model.Nodes) +      
                  sum(
                    node[3,n]['prob'] *  
                    params['VFfac'] * VF(node[3,n]['price_level'], model.d[3,n], model.l[3,n]) 
                    for n in nodes_in(3)), # added VF
            sense= maximize)
    else:
        model.obj = Objective(
            expr= sum( 
                    node[tupl]['prob'] * (model.rho_up[tupl] - model.rho_dw[tupl] - model.c[tupl]) 
                    for tupl in model.Nodes),
            sense= maximize)

    
    
    # Objective (paths)
#    if useVF:
#        model.obj = Objective(
#            expr= sum( 
#                    node[3,n]['prob'] * 
#                        (model.rho_up[3,n]               - model.rho_dw[3,n]               - model.c[3,n] + 
#                         model.rho_up[2,anc(3,n)]        - model.rho_dw[2,anc(3,n)]        - model.c[2,anc(3,n)] +
#                         model.rho_up[1,anc(2,anc(3,n))] - model.rho_dw[1,anc(2,anc(3,n))] - model.c[1,anc(2,anc(3,n))] +
#                         model.rho_up[0,0] - model.rho_dw[0,0] - model.c[0,0]) 
#                    for n in nodes_in(3)) +      
#                   sum(
#                    node[3,n]['prob'] *
#                    params['VFfac'] * VF(node[3,n]['price_level'], model.d[3,n], model.l[3,n]) 
#                    for n in nodes_in(3)), # added VF
#            sense= maximize)
#    else:
#        model.obj = Objective(
#            expr= sum( 
#                    node[3,n]['prob'] * 
#                        (model.rho_up[3,n]               - model.rho_dw[3,n]               - model.c[3,n] + 
#                         model.rho_up[2,anc(3,n)]        - model.rho_dw[2,anc(3,n)]        - model.c[2,anc(3,n)] +
#                         model.rho_up[1,anc(2,anc(3,n))] - model.rho_dw[1,anc(2,anc(3,n))] - model.c[1,anc(2,anc(3,n))] +
#                         model.rho_up[0,0]               - model.rho_dw[0,0]               - model.c[0,0]) 
#                    for n in nodes_in(3)),
#            sense= maximize)
    
    
    

    # Constraints
    # Node energy balance
    model.node_energy_balance = ConstraintList()
    for tupl in model.Nodes:
        model.node_energy_balance.add(
            q_DA[tupl[0]] + model.q_up[tupl] - model.q_dw[tupl] == model.d[tupl] - model.p_up[tupl] + 
                                                                   model.p_dw[tupl] + 
                                                                   model.vi_plus[tupl] - model.vi_minus[tupl]  
        )

    # Convert offer curves of stages 0,1 and 2 to piecewise functions
    offer_curves_as_fcn = {
        0: {
            'up': offer_curve_as_fcn(0,'up', offerCurves), 
            'dw': offer_curve_as_fcn(0,'dw', offerCurves)
        },
        1: {
            'up': offer_curve_as_fcn(1,'up', offerCurves), 
            'dw': offer_curve_as_fcn(1,'dw', offerCurves)
        },
        2: {
            'up': offer_curve_as_fcn(2,'up', offerCurves), 
            'dw': offer_curve_as_fcn(2,'dw', offerCurves)
        }
    }
    
    # Constrain production in stages 0,1 and 2 by offer curves submitted in the past    
    model.off_curves = ConstraintList()
    for tupl in model.Nodes:
        if tupl[0] < 3:
            model.off_curves.add( model.q_up[tupl] == offer_curves_as_fcn[tupl[0]]['up'](node[tupl]['price']))
            model.off_curves.add( model.q_dw[tupl] == offer_curves_as_fcn[tupl[0]]['dw'](node[tupl]['price']))


    # Match quantities (q's) with incremental quantities (o's)
    model.q_const = ConstraintList()
    for t in model.Stages:  
        for n in nodes_in(t):
            model.q_const.add(
                model.q_up[t,n] == sum( M_up(n, n_, t) * model.o_up[t,n_] for n_ in nodes_in(t))
            )  
            model.q_const.add(
                model.q_dw[t,n] == sum( M_dw(n, n_, t) * model.o_dw[t,n_] for n_ in nodes_in(t))
            )
    

    # Calculate profit 
    model.rho_const = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            model.rho_const.add(
                model.rho_up[t,n] == sum( M_up(n,n_,t) * node[t,n_]['price'] * model.o_up[t,n_] for n_ in nodes_in(t))
            )
            model.rho_const.add(
                model.rho_dw[t,n] == sum( M_dw(n,n_,t) * node[t,n_]['price'] * model.o_dw[t,n_] for n_ in nodes_in(t))
            )


    # Offer constraints: check if market requires up/dw regulation
    model.offer_curves = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            if node[t,n]['price'] <= price_DA[t]:
                model.offer_curves.add( model.o_up[t,n] == 0 )
            elif node[t,n]['price'] >= price_DA[t]:
                model.offer_curves.add( model.o_dw[t,n] == 0 )

    
    # Operational constraints
    # Storage
    model.storage_const = ConstraintList()
    for t in model.Stages:
        # fix 'L0'
        if t == 0:
            for n in nodes_in(t):
                model.storage_const.add(
                    model.l[t,n] == L0 + params['ETA'] * model.p_up[t,n] - model.p_dw[t,n]
                )    
        else:
            for n in nodes_in(t):
                model.storage_const.add(
                    model.l[t,n] == model.l[t-1,anc(t,n)] + params['ETA'] * model.p_up[t,n] - model.p_dw[t,n]
                )

        # fix 'l' at the end of stage 3 for every node in stage 3
        #for n in nodes_in(t):
        #    if t != 3 and storageIn[t]['free'] == False:
        #        if t == 0:
        #            dsdsds
        #        else:
        #            model.storage_const.add( model.l[t,n] == storageIn[node[t,n]['price_level']] )
    
        #if t == 3 and storageIn['free'] == False:
        #    for n in nodes_in(t):
        #        model.storage_const.add(model.l[t,n] == storageIn[node[t,n]['price_level']] )

        for n in nodes_in(t):
            model.storage_const.add( model.l[t,n] <= params['LMAX'])
            model.storage_const.add( model.l[t,n] >= params['LMIN'])
            model.storage_const.add( model.p_up[t,n] <= params['PUP'])
            model.storage_const.add( model.p_dw[t,n] <= params['PDW'])

        
    # Thermal unit
    model.thermal_const = ConstraintList()
    for t in model.Stages:
        if t == 0:
            for n in nodes_in(t):
                model.thermal_const.add(model.d[t,n] - D0 <= params['DRAMP'])
                model.thermal_const.add(D0 - model.d[t,n] <= params['DRAMP'])
                model.thermal_const.add(model.u[t,n] - params['U0'] <= model.y[t,n])
                model.thermal_const.add(params['U0'] - model.u[t,n] <= model.z[t,n])
        else:
            for n in nodes_in(t):
                model.thermal_const.add(model.d[t,n] - model.d[t-1,anc(t,n)] <= params['DRAMP'])
                model.thermal_const.add(model.d[t-1,anc(t,n)] - model.d[t,n] <= params['DRAMP'])
                model.thermal_const.add(model.u[t,n] - model.u[t-1,anc(t,n)] <= model.y[t,n])
                model.thermal_const.add(model.u[t-1,anc(t,n)] - model.u[t,n] <= model.z[t,n])

        for n in nodes_in(t):
            model.thermal_const.add( model.u[t,n] * params['DMIN'] <= model.d[t,n] )
            model.thermal_const.add( model.u[t,n] * params['DMAX'] >= model.d[t,n] )
    

    # Cost constraints
    model.cost = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            
            if t==0:
                model.cost.add(
                    params['C0']*model.u[t,n] + params['MC']*model.d[t,n] + params['CST']*model.y[t,n] + 
                    params['CSH']*model.z[t,n] +
                    params['MST']*(model.vi_plus[t,n] + model.vi_minus[t,n]) 
                    == model.c[t,n] 
                )
            else:
                model.cost.add(
                    params['C0']*model.u[t,n] + params['MC']*model.d[t,n] + params['CST']*model.y[t,n] + 
                    params['CSH']*model.z[t,n] +
                    params['MST']*(model.vi_plus[t,n] + model.vi_minus[t,n]) 
                    == model.c[t,n] 
                )
    
    # Fix binary variables
    model.fix = ConstraintList()
    for tupl in model.Nodes:
        model.fix.add(model.u[tupl]==1)
        model.fix.add(model.y[tupl]==1)
        model.fix.add(model.z[tupl]==1)
            
        

     
    

            
    ###########################
    # Solve Three-stage model #
    ###########################        
    solver_parameters = "ResultFile=model.ilp"        
    solver = SolverFactory('gurobi_ampl', symbolic_solver_labels=True) 
    results = solver.solve(model, options_string=solver_parameters, tee=False) # add: tee=True to print solver stats
    
    print(" ")
    print("Solver status:", results.solver.status)
    print("Problem condition: ", results.solver.termination_condition)
    print("Time: ", round(results.solver.time,3), "secs.")
    
    if (results.solver.status == SolverStatus.ok) and \
       (results.solver.termination_condition == TerminationCondition.optimal):
        print("Problem is Ok and feasible.")
        
        # Offer curves: (price, quantity) pairs.
        x_up = [model.q_up[3,n]() for n in nodes_in(3)] 
        x_dw = [model.q_dw[3,n]() for n in nodes_in(3)] 
        price = np.round([node[3,n]['price'] for n in nodes_in(3)],4) 

        pairs_up = []
        pairs_dw = []
        for n in nodes_in(3):
            # (price, quantity)
            pairs_up.append((node[3,n]['price'],model.q_up[3,n]()))
            pairs_dw.append((node[3,n]['price'],model.q_dw[3,n]()))   # to model
            
        offer_curves_submit = {
            'up': {pair[0]: pair[1] for pair in pairs_up}, 
            'dw': {pair[0]: pair[1] for pair in pairs_dw},
        }
        
        
        # Calculate profit in whole stage
        print("Solution for hour {}".format(h))
        objval = 0
        if useVF == True:
            sumVF = sum([node[3,s]['prob'] * VF(node[3,s]['price_level'],model.d[3,s](), model.l[3,s]()) 
                        for s in nodes_in(3)])
            objval = model.obj() - sumVF
            if log: print("Profit: {}, sumVF: {}".format(objval,sumVF))
        else:
            objval = model.obj()
            if log: print("Profit: {}".format(objval))
            
            
            
        # Storage level by the end of stage 3
        #storageLevel = {'free': False}
        #for n in nodes_in(3):
        #    if n <= Ns:
        #        storageLevel.update({
        #                node[3,n]['price_level']: model.l[3,n]()
        #        })
        
        
        # Return model.l[0,0]
        newL0 = model.l[0,0]()
        
        # Return model.d[0,0]
        newD0 = model.d[0,0]()
        
        # Return profit in stage 0
        λ = node[0,0]['price']
        profUp=0
        profDw=0
        if λ > price_DA[0]:
            for p,q in offerCurves[0]['up'].items():
                if λ >= p:
                    profUp+= p*q
        
        elif λ < price_DA[0]:
            for p,q in offerCurves[0]['dw'].items():
                if λ <= p:
                    profDw+= p*q
        
        profit0 = [price_DA[0]*q_DA[0], profUp, profDw, model.c[0,0]()]
        if log: print("Profit hour {} = {} ".format(h, profit0[0] + profit0[1] - profit0[2] - profit0[3]))
    
    
        # Check if penalty is incurred  
        # penal[0] = 0/1 if there was a penalty in any of the stages
        # penal[1] = 0/1 if there was a penalty in stage 0
        penal = [0,0,0,0]        
        for t in model.Stages:
            for n in nodes_in(t):
                if model.vi_plus[t,n]() >= 0.01:
                    if log: print("vi_plus[{},{}]={}".format(t,n,model.vi_plus[t,n]()))
                    penal[t] = 1
                if model.vi_minus[t,n]() >= 0.01:
                    if log: print("vi_minus[{},{}]={}".format(t,n,model.vi_minus[t,n]()))
                    penal[t] = 1

        
        
        #  Return
        if log:
            print("BM price hour {}: {}".format(h,node[0,0]['price']))
            print("DA price hour {}: {}".format(h, price_DA[0]))
            print("qda={}, qup={}, qdw={}".format(q_DA[0], model.q_up[0,0](),model.q_dw[0,0]()))
            print("D0={}, L0={}".format(D0, L0))
            print("d={}, l={}, pup={}, pdw={}".format(model.d[0,0](), model.l[0,0](), model.p_up[0,0](), model.p_dw[0,0]()))
 
            #print(model.display())
            
        return(offer_curves_submit, newL0, newD0, objval, profit0, penal)

    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print("Problem is infeasible.")
        return(solver,results,0,0,0,0)
    else:
        print("Something went wrong.")
        print("Solver Status: ",  results.solver.status)
        return(solver,results,0,0,0,0)
    

def runThreeStageBorder(hour, params, data, probMat, offerCurves, L0, D0, CC, useVF=False, log=False):
    ''' Run Three-stage stochastic model for the three last hours of the day.
    Inputs:
    hour: current hour
    data: price scenarios (Bm_dict) - [dict]
    probMat: node transition probabilities (Prob_ay) - [np.array]
    
    Output:
    Offer curve for time 'hour+3'
    
    '''

    ###########################
    # Initial Setup          #
    ##########################
    
    # Scenarios Scheme
    # One-stage decision: go from node (0,0) to nodes in stage 1,2 and 3.
    # Nodes are identified as node[t,n], time stage t and id n.

    # Parameters
    h = hour
    
    #T = 3                                   # number of time/stages [0-3]
    T = 0
    if h == 22:
        T = 2
    elif h == 23:
        T = 1
    elif h == 24:
        T = 0
    
    Ns = len(probMat)                       # number of price scenarios per stage
    N = sum([Ns**i for i in range(1,T+1)])  # number of nodes (w/o counting node (0,0))
    #S = Ns ** T                            # number of scenario paths


    
    
    # Node tree 
    node = {}

    # t=0
    node.update({
        (0,0): {
            'price': data['hour{0}'.format(h)]['lambda_bm_scend'][('h0','j1')],
            'prob': 1,
            'ancestor': None,
        }
    })

    # t=1,2,3
    for t in range(1,T+1):
        for n in range(1,Ns**t + 1):
            j = n%Ns if n%Ns != 0 else Ns    # scenario mapped to multiples of Ns
            i = (n-1)//Ns + 1 if t!=1 else 0 # ancestor
            i_map = i%Ns if i%Ns != 0 else Ns

            node.update({
                (t,n): {
                    'price': data['hour{0}'.format(h)]['lambda_bm_scend'][('h{0}'.format(t),'j{0}'.format(j))], 
                    'tprob' : probMat[i_map-1][j-1] if t!=1 else 1.0/Ns , # transition prob. from last state
                    'ancestor': (t-1,i),
                    'price_level': j}
            })

    # Set node's absolute probabilities
    for t in range(1,T+1):
        for n in range(1,Ns**t+1):
            if t == 1:
                node[t,n]['prob'] = node[t,n]['tprob']
            else:
                node[t,n]['prob'] = node[t,n]['tprob'] * node[node[t,n]['ancestor']]['prob']


    # Day-ahead
    #price_DA = {
    #    0: data['hour{0}'.format(h  )]['lambda_da_real'],
    #    1: data['hour{0}'.format(h+1)]['lambda_da_real'],
    #    2: data['hour{0}'.format(h+2)]['lambda_da_real'],
    #    3: data['hour{0}'.format(h+3)]['lambda_da_real'],
    #}

    # Day-ahead
    price_DA = { t: data['hour{0}'.format(h+t)]['lambda_da_real'] for t in range(T+1)}


    q_DA = {
        0: params['QDA'], # generate now, including regulation
        1: params['QDA'],
        2: params['QDA'],
        3: params['QDA'],
    }
 

    # Local functions
    def M_up(n, n_, t):
        if node[t,n]['price'] >= node[t,n_]['price'] and node[t,n]['price'] > price_DA[t]:
            return 1
        else:
            return 0

    def M_dw(n, n_, t):
        if node[t,n]['price'] <= node[t,n_]['price'] and node[t,n]['price'] < price_DA[t]:
            return 1
        else:
            return 0

    def anc(t,n):
        return node[t,n]['ancestor'][1]

    def nodes_in(t):
        # return id of nodes in stage t
        ret = []
        for i,tupl in enumerate(node.keys()):
            if tupl[0] == t:
                ret.append(tupl[1])
        return ret

    def offer_curve_as_fcn(t, reg, off_curves):
        # outputs a piecewise function
        # the output of the function is the cleared 'quantity' for a 'price' input
        p,q=0,0
        if reg == 'up':
            sorted_offer_curve = {k: off_curves[t][reg][k] for k in sorted(off_curves[t][reg].keys())}
            p = list(sorted_offer_curve.keys())
            q = list(sorted_offer_curve.values())
        elif reg == 'dw':
            sorted_offer_curve = {k: off_curves[t][reg][k] for k in sorted(off_curves[t][reg].keys())}
            p = list(sorted_offer_curve.keys())
            p.insert(0,0)
            q = list(sorted_offer_curve.values())
            q.append(0)
        def fcn(x):
            return float(np.piecewise(x, [x >= num for num in p], q))
        return fcn
    
    
    ###########################
    # Build Three-stage model #
    ###########################
    
    # Pyomo Model
    model = ConcreteModel()
    model.Stages = range(0,T+1)
    model.NodesId = range(0,N+1+1) # include node (0,0)
    model.Nodes = node.keys()

    # Variables
    model.o_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.o_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.q_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.q_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.rho_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,10000000))
    model.rho_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,10000000))

    model.d = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.l = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.c = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000000))
    model.p_up = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.p_dw = Var(model.Nodes,within=NonNegativeReals,bounds=(0,1000))
    model.u = Var(model.Nodes,within=Binary)
    model.y = Var(model.Nodes,within=Binary)
    model.z = Var(model.Nodes,within=Binary)
    
    model.vi_plus  = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000)) # virtual generator / penalty
    model.vi_minus = Var(model.Nodes,within=NonNegativeReals,bounds=(0,100000))
    
    # Value function (continuous)
    def VF(s,x,y):
        return(CC[s][0] + CC[s][1]*x + CC[s][2]*y + CC[s][3]*x*y + CC[s][4]*x**2 + CC[s][5]*y**2)
        
        
    # Objective (nodes)
    model.obj = Objective(
        expr= sum( 
                node[tupl]['prob'] * (model.rho_up[tupl] - model.rho_dw[tupl] - model.c[tupl]) 
                for tupl in model.Nodes),
        sense= maximize)

    
    
    # Objective (paths)
#    if useVF:
#        model.obj = Objective(
#            expr= sum( 
#                    node[3,n]['prob'] * 
#                        (model.rho_up[3,n]               - model.rho_dw[3,n]               - model.c[3,n] + 
#                         model.rho_up[2,anc(3,n)]        - model.rho_dw[2,anc(3,n)]        - model.c[2,anc(3,n)] +
#                         model.rho_up[1,anc(2,anc(3,n))] - model.rho_dw[1,anc(2,anc(3,n))] - model.c[1,anc(2,anc(3,n))] +
#                         model.rho_up[0,0] - model.rho_dw[0,0] - model.c[0,0]) 
#                    for n in nodes_in(3)) +      
#                   sum(
#                    node[3,n]['prob'] *
#                    params['VFfac'] * VF(node[3,n]['price_level'], model.d[3,n], model.l[3,n]) 
#                    for n in nodes_in(3)), # added VF
#            sense= maximize)
#    else:
#        model.obj = Objective(
#            expr= sum( 
#                    node[3,n]['prob'] * 
#                        (model.rho_up[3,n]               - model.rho_dw[3,n]               - model.c[3,n] + 
#                         model.rho_up[2,anc(3,n)]        - model.rho_dw[2,anc(3,n)]        - model.c[2,anc(3,n)] +
#                         model.rho_up[1,anc(2,anc(3,n))] - model.rho_dw[1,anc(2,anc(3,n))] - model.c[1,anc(2,anc(3,n))] +
#                         model.rho_up[0,0]               - model.rho_dw[0,0]               - model.c[0,0]) 
#                    for n in nodes_in(3)),
#            sense= maximize)
    
    
    

    # Constraints
    # Node energy balance
    model.node_energy_balance = ConstraintList()
    for tupl in model.Nodes:
        model.node_energy_balance.add(
            q_DA[tupl[0]] + model.q_up[tupl] - model.q_dw[tupl] == model.d[tupl] - model.p_up[tupl] + 
                                                                   model.p_dw[tupl] + 
                                                                   model.vi_plus[tupl] - model.vi_minus[tupl]  
        )

    # Convert offer curves of stages 0,1 and 2 to piecewise functions
    offer_curves_as_fcn = {
        0: {
            'up': offer_curve_as_fcn(0,'up', offerCurves), 
            'dw': offer_curve_as_fcn(0,'dw', offerCurves)
        },
        1: {
            'up': offer_curve_as_fcn(1,'up', offerCurves), 
            'dw': offer_curve_as_fcn(1,'dw', offerCurves)
        },
        2: {
            'up': offer_curve_as_fcn(2,'up', offerCurves), 
            'dw': offer_curve_as_fcn(2,'dw', offerCurves)
        }
    }
    
    # Constrain production in stages 0,1 and 2 by offer curves submitted in the past    
    model.off_curves = ConstraintList()
    for tupl in model.Nodes:
        if tupl[0] < 3:
            model.off_curves.add( model.q_up[tupl] == offer_curves_as_fcn[tupl[0]]['up'](node[tupl]['price']))
            model.off_curves.add( model.q_dw[tupl] == offer_curves_as_fcn[tupl[0]]['dw'](node[tupl]['price']))


    # Match quantities (q's) with incremental quantities (o's)
    model.q_const = ConstraintList()
    for t in model.Stages:  
        for n in nodes_in(t):
            model.q_const.add(
                model.q_up[t,n] == sum( M_up(n, n_, t) * model.o_up[t,n_] for n_ in nodes_in(t))
            )  
            model.q_const.add(
                model.q_dw[t,n] == sum( M_dw(n, n_, t) * model.o_dw[t,n_] for n_ in nodes_in(t))
            )
    

    # Calculate profit 
    model.rho_const = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            model.rho_const.add(
                model.rho_up[t,n] == sum( M_up(n,n_,t) * node[t,n_]['price'] * model.o_up[t,n_] for n_ in nodes_in(t))
            )
            model.rho_const.add(
                model.rho_dw[t,n] == sum( M_dw(n,n_,t) * node[t,n_]['price'] * model.o_dw[t,n_] for n_ in nodes_in(t))
            )


    # Offer constraints: check if market requires up/dw regulation
    model.offer_curves = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            if node[t,n]['price'] <= price_DA[t]:
                model.offer_curves.add( model.o_up[t,n] == 0 )
            elif node[t,n]['price'] >= price_DA[t]:
                model.offer_curves.add( model.o_dw[t,n] == 0 )

    
    # Operational constraints
    # Storage
    model.storage_const = ConstraintList()
    for t in model.Stages:
        # fix 'L0'
        if t == 0:
            for n in nodes_in(t):
                model.storage_const.add(
                    model.l[t,n] == L0 + params['ETA'] * model.p_up[t,n] - model.p_dw[t,n]
                )    
        else:
            for n in nodes_in(t):
                model.storage_const.add(
                    model.l[t,n] == model.l[t-1,anc(t,n)] + params['ETA'] * model.p_up[t,n] - model.p_dw[t,n]
                )

        # fix 'l' at the end of stage 3 for every node in stage 3
        #for n in nodes_in(t):
        #    if t != 3 and storageIn[t]['free'] == False:
        #        if t == 0:
        #            dsdsds
        #        else:
        #            model.storage_const.add( model.l[t,n] == storageIn[node[t,n]['price_level']] )
    
        #if t == 3 and storageIn['free'] == False:
        #    for n in nodes_in(t):
        #        model.storage_const.add(model.l[t,n] == storageIn[node[t,n]['price_level']] )

        for n in nodes_in(t):
            model.storage_const.add( model.l[t,n] <= params['LMAX'])
            model.storage_const.add( model.l[t,n] >= params['LMIN'])
            model.storage_const.add( model.p_up[t,n] <= params['PUP'])
            model.storage_const.add( model.p_dw[t,n] <= params['PDW'])

        
    # Thermal unit
    model.thermal_const = ConstraintList()
    for t in model.Stages:
        if t == 0:
            for n in nodes_in(t):
                model.thermal_const.add(model.d[t,n] - D0 <= params['DRAMP'])
                model.thermal_const.add(D0 - model.d[t,n] <= params['DRAMP'])
                model.thermal_const.add(model.u[t,n] - params['U0'] <= model.y[t,n])
                model.thermal_const.add(params['U0'] - model.u[t,n] <= model.z[t,n])
        else:
            for n in nodes_in(t):
                model.thermal_const.add(model.d[t,n] - model.d[t-1,anc(t,n)] <= params['DRAMP'])
                model.thermal_const.add(model.d[t-1,anc(t,n)] - model.d[t,n] <= params['DRAMP'])
                model.thermal_const.add(model.u[t,n] - model.u[t-1,anc(t,n)] <= model.y[t,n])
                model.thermal_const.add(model.u[t-1,anc(t,n)] - model.u[t,n] <= model.z[t,n])

        for n in nodes_in(t):
            model.thermal_const.add( model.u[t,n] * params['DMIN'] <= model.d[t,n] )
            model.thermal_const.add( model.u[t,n] * params['DMAX'] >= model.d[t,n] )
    

    # Cost constraints
    model.cost = ConstraintList()
    for t in model.Stages:
        for n in nodes_in(t):
            
            if t==0:
                model.cost.add(
                    params['C0']*model.u[t,n] + params['MC']*model.d[t,n] + params['CST']*model.y[t,n] + 
                    params['CSH']*model.z[t,n] +
                    params['MST']*(model.vi_plus[t,n] + model.vi_minus[t,n]) 
                    == model.c[t,n] 
                )
            else:
                model.cost.add(
                    params['C0']*model.u[t,n] + params['MC']*model.d[t,n] + params['CST']*model.y[t,n] + 
                    params['CSH']*model.z[t,n] +
                    params['MST']*(model.vi_plus[t,n] + model.vi_minus[t,n]) 
                    == model.c[t,n] 
                )
    
    # Fix binary variables
    model.fix = ConstraintList()
    for tupl in model.Nodes:
        model.fix.add(model.u[tupl]==1)
        model.fix.add(model.y[tupl]==1)
        model.fix.add(model.z[tupl]==1)
            
        

     
    

            
    ###########################
    # Solve Three-stage model #
    ###########################        
    solver_parameters = "ResultFile=model.ilp"        
    solver = SolverFactory('gurobi_ampl', symbolic_solver_labels=True) 
    results = solver.solve(model, options_string=solver_parameters, tee=False) # add: tee=True to print solver stats
    
    print(" ")
    print("Solver status:", results.solver.status)
    print("Problem condition: ", results.solver.termination_condition)
    print("Time: ", round(results.solver.time,3), "secs.")
    
    if (results.solver.status == SolverStatus.ok) and \
       (results.solver.termination_condition == TerminationCondition.optimal):
        print("Problem is Ok and feasible.")
        
        # Offer curves: (price, quantity) pairs.
        #x_up = [model.q_up[3,n]() for n in nodes_in(3)] 
        #x_dw = [model.q_dw[3,n]() for n in nodes_in(3)] 
        #price = np.round([node[3,n]['price'] for n in nodes_in(3)],4) 
        x_up = [0,0,0] 
        x_dw = [0,0,0] 
        price = [0,0,0] 

        #pairs_up = []
        #pairs_dw = []
        #for n in nodes_in(3):
        #    # (price, quantity)
        #    pairs_up.append((node[3,n]['price'],model.q_up[3,n]()))
        #    pairs_dw.append((node[3,n]['price'],model.q_dw[3,n]()))   # to model
        pairs_up = []
        pairs_dw = []
        for n in range(3):
            # (price, quantity)
            pairs_up.append((price[n],x_up[n]))
            pairs_dw.append((price[n],x_dw[n]))
            
        offer_curves_submit = {
            'up': {pair[0]: pair[1] for pair in pairs_up}, 
            'dw': {pair[0]: pair[1] for pair in pairs_dw},
        }
        
        
        # Calculate profit in whole stage
        print("Solution for hour {}".format(h))
        objval = 0
        if useVF == True:
            sumVF = sum([node[3,s]['prob'] * VF(node[3,s]['price_level'],model.d[3,s](), model.l[3,s]()) 
                        for s in nodes_in(3)])
            objval = model.obj() - sumVF
            if log: print("Profit: {}, sumVF: {}".format(objval,sumVF))
        else:
            objval = model.obj()
            if log: print("Profit: {}".format(objval))
            
            
            
        # Storage level by the end of stage 3
        #storageLevel = {'free': False}
        #for n in nodes_in(3):
        #    if n <= Ns:
        #        storageLevel.update({
        #                node[3,n]['price_level']: model.l[3,n]()
        #        })
        
        
        # Return model.l[0,0]
        newL0 = model.l[0,0]()
        
        # Return model.d[0,0]
        newD0 = model.d[0,0]()
        
        # Return profit in stage 0
        λ = node[0,0]['price']
        profUp=0
        profDw=0
        if λ > price_DA[0]:
            for p,q in offerCurves[0]['up'].items():
                if λ >= p:
                    profUp+= p*q
        
        elif λ < price_DA[0]:
            for p,q in offerCurves[0]['dw'].items():
                if λ <= p:
                    profDw+= p*q
        
        profit0 = [price_DA[0]*q_DA[0], profUp, profDw, model.c[0,0]()]
        if log: print("Profit hour {} = {} ".format(h, profit0[0] + profit0[1] - profit0[2] - profit0[3]))
    
    
        # Check if penalty is incurred  
        # penal[0] = 0/1 if there was a penalty in any of the stages
        # penal[1] = 0/1 if there was a penalty in stage 0
        penal = [0,0,0,0]        
        for t in model.Stages:
            for n in nodes_in(t):
                if model.vi_plus[t,n]() >= 0.01:
                    if log: print("vi_plus[{},{}]={}".format(t,n,model.vi_plus[t,n]()))
                    penal[t] = 1
                if model.vi_minus[t,n]() >= 0.01:
                    if log: print("vi_minus[{},{}]={}".format(t,n,model.vi_minus[t,n]()))
                    penal[t] = 1

        
        
        #  Return
        if log:
            print("BM price hour {}: {}".format(h,node[0,0]['price']))
            print("DA price hour {}: {}".format(h, price_DA[0]))
            print("qda={}, qup={}, qdw={}".format(q_DA[0], model.q_up[0,0](),model.q_dw[0,0]()))
            print("D0={}, L0={}".format(D0, L0))
            print("d={}, l={}, pup={}, pdw={}".format(model.d[0,0](), model.l[0,0](), model.p_up[0,0](), model.p_dw[0,0]()))
 
            #print(model.display())
            
        return(offer_curves_submit, newL0, newD0, objval, profit0, penal)

    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print("Problem is infeasible.")
        return(solver,results,0,0,0,0)
    else:
        print("Something went wrong.")
        print("Solver Status: ",  results.solver.status)
        return(solver,results,0,0,0,0)
    

def plotOfferCurves(offerCurves, hour, data):
    pairs_up = [(k,v) for k,v in offerCurves['up'].items()]
    pairs_dw = [(k,-v) for k,v in offerCurves['dw'].items()]
    
    DAprice = data['hour{0}'.format(hour+3)]['lambda_da_real']
    
    pairs_up.insert(0,(DAprice, 0)) # insert DA pair at the beginning
    pairs_dw.append((DAprice, 0))   # insert DA pair at the end
    
    pairs_up = sorted(pairs_up)
    pairs_dw = sorted(pairs_dw)
    
    # try fix no-ending curve
    pairs_up.insert(len(pairs_up),(pairs_up[len(pairs_up)-1][0]+1, pairs_up[len(pairs_up)-1][1]))
    pairs_dw.insert(0,(pairs_dw[0][0]-1, pairs_dw[0][1]))
    
    plt.step(list(zip(*pairs_up))[0],list(zip(*pairs_up))[1], where='post',label=r'up. reg.') # pre is default
    plt.step(list(zip(*pairs_dw))[0],list(zip(*pairs_dw))[1], where='pre', label=r'dw. reg.',linestyle="dashed")
    plt.axvline(x=DAprice, label='Day-ahead price', ls='dotted', color='gray', linewidth=2.5)
    plt.ylabel("quantity[MWh]", fontsize=15)
    plt.xlabel("price[MWh]", fontsize=15)
    #plt.title("Stage 3 Offering Curves")
    plt.grid(c='lightgray',ls='dotted')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xlim(0,100)  
    #plt.ylim(-100,100)
    #plt.yticks(np.arange(-100, 101, 10))
    #plt.legend()
    #plt.figure(figsize=(10,10))
    plt.show()
    
