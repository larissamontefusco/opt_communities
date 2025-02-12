import numpy as np
import pyomo.environ as pe
import pyomo.opt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from pyomo.util.infeasible import log_infeasible_constraints
import os
import pandas as pd
import dotenv

# Functions to the model
# Set default behaviour
default_behaviour = pe.Constraint.Skip

# Auxiliary function to convert numpy arrays to dictionaries
def convert_to_dictionary(a, t_start=0):
    temp_dictionary = {}

    if len(a.shape) == 3:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                for dim2 in np.arange(a.shape[2]):
                    temp_dictionary[(dim0+1, dim1+1, dim2+1+t_start)] = a[dim0, dim1, dim2]
    elif len(a.shape) == 2:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                temp_dictionary[(dim0+1, dim1+1+t_start)] = a[dim0, dim1]

    else:
        for dim0 in np.arange(a.shape[0]):
            temp_dictionary[(dim0+1+t_start)] = a[dim0]

    return temp_dictionary

data_geral = pd.read_excel("./excel_task_newopt.xlsx", sheet_name="GeneralInfo", header=None)
data_gen = pd.read_excel("./excel_task_newopt.xlsx", sheet_name="Gen", header=None)
data_load = pd.read_excel("./excel_task_newopt.xlsx", sheet_name="Load", header=None)
data_bess = pd.read_excel("./excel_task_newopt.xlsx", sheet_name="BESS", header=None)
data_evs = pd.read_excel("./excel_task_newopt.xlsx", sheet_name="EVS", header=None)

def get_timeseries(values, component: str):

    temp_data_idx = np.where(values == component)
    temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
    for i in np.arange(temp_data.shape[0]):
        temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
    temp_data = temp_data[:, temp_data_idx[1][0] + 1:]

    return temp_data

def get_characteristic(values, component: str, keep_string: bool = False):

    temp_data_idx = np.where(values == component)
    temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
    if not keep_string:
        for i in np.arange(temp_data.shape[0]):
            temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
        temp_data = temp_data[:, temp_data_idx[1][0] + 1]
    else:
        for i in np.arange(temp_data.shape[0]):
            temp_data[i, :] = temp_data[i, :].astype(str)
        temp_data = temp_data[:, temp_data_idx[1][0] + 1]

    return temp_data



def params(model):
    model.impMax = get_characteristic(data_geral, 'Imp Max (kW)')               
    model.expMax = get_characteristic(data_geral, 'Exp Max (kW)')
    
    # GENERATORS:
    model.gen = pe.Set(initialize=np.arange(1, len(get_characteristic(data_gen, 'Nominal Power (kW)')) + 1),
                        doc='Number of generators')
    model.genMax = pe.Param(model.gen, initialize=convert_to_dictionary(get_characteristic(data_gen, 'Nominal Power (kW)')),
                                doc='Gen Max')
    
    model.genValues = pe.Param(model.gen, model.t,
                            initialize=convert_to_dictionary(get_timeseries(data_gen, 'Production (kW)')),
                            doc='Forecasted power generation')
    model.genType = pe.Param(model.gen,
                            initialize=convert_to_dictionary(get_characteristic(data_gen, 'Gen Type Id')),
                            doc='Types of generators')
    # @TODO O Manuel não me mandou nenhum dado de preço de exportação, importação, etc
    model.genActCost = 0.08
    model.genExcCost = 0 
    
    # LOADS:
    model.loads = pe.Set(initialize=np.arange(1, len(get_characteristic(data_load, 'Nominal Power (kW)')) + 1),doc='Number of loads')
    
    #@TODO NÃO TO A USAR O LOAD MAX PRA NADA POR ENQUANTO
#    model.loadMax = pe.Param(model.loads, initialize=convert_to_dictionary(get_characteristic(data_load, 'Nominal Power (kW)')),
#                                doc='Load Max')
    model.loadValues = pe.Param(model.loads, model.t, initialize=convert_to_dictionary(get_timeseries(data_load, 'Consumption (kW)')),
                                doc='Load Power During time')
    # GRID IMPORT AND EXPORT:
    model.impCost = 1
    model.expCost = 1

    # BESS:
    model.stor = pe.Set(initialize=np.arange(1, len(get_characteristic(data_bess, 'Nominal Power (kW)')) + 1), doc='Number of storage units')

    model.storMax = pe.Param(model.stor, initialize=convert_to_dictionary(get_characteristic(data_bess, 'Nominal Power (kW)'))) #kW

    model.storDchMax = pe.Param(model.stor,
                                initialize=convert_to_dictionary(get_characteristic(data_bess, 'Maximum Power Delivery (kW)')),
                                doc='Starting energy capacity')
    model.storChMax = pe.Param(model.stor,
                                initialize=convert_to_dictionary(get_characteristic(data_bess, 'Maximum Power Charge (kW)')),
                                doc='Starting energy capacity')

    model.storMin = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Lower SOC limit (kW)')),
                            doc='Starting energy capacity')
    model.initial_soc = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Initial SOC (%)')/100),
                            doc='Starting SoC (%)')

    model.storStart = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Initial SOC (%)')/100 * 
                                                            get_characteristic(data_bess, 'Nominal Power (kW)')))

    model.storDchEff = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Discharge Efficiency (%)')/100),
                            doc='Starting SoC (%)')
    model.storChEff = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Charge Efficiency (%)')/100),
                            doc='Starting SoC (%)')
    model.storBatteryTarget = pe.Param(model.stor,
                            initialize=convert_to_dictionary(get_characteristic(data_bess, 'Target (%)')/100),
                            doc='Target SoC (%)')
    model.storChCost = 1
    model.storDchCost = 1
    
    # EVS:
    model.v2g = pe.Set(initialize=np.arange(1, len(get_characteristic(data_evs, 'Maximum Power Delivery (kW)')) + 1),
                        doc='Number of EVs')
    model.v2gDchMax = pe.Param(model.v2g, 
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Maximum Power Delivery (kW)')),
                            doc='Maximum scheduled discharging power')
    model.v2gChMax = pe.Param(model.v2g, 
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Maximum Power Charge (kW)')),
                            doc='Maximum scheduled charging power')

    model.v2gDchEff = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Discharge Efficiency (%)')/100),
                            doc='Discharging efficiency')
    model.v2gChEff = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Charge Efficiency (%)')/100),
                            doc='Charging efficiency')
    model.v2gMax = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Nominal Power (kW)')),
                            doc='Maximum energy capacity')
    model.v2gMin = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Lower SOC limit (kW)')),
                            doc='Minimum energy capacity')
    model.v2gConnected = pe.Param(model.v2g, model.t,
                                initialize=convert_to_dictionary(get_timeseries(data_evs, 'Connected')),
                                doc='Vehicle schedule')

    model.v2gScheduleArrivalSOC = pe.Param(model.v2g,
                                        initialize=convert_to_dictionary(get_characteristic(data_evs, 'Initial SOC (%)')/100),
                                        doc='Vehicle schedule arrival SOC')
    model.v2gScheduleDepartureSOC = pe.Param(model.v2g,
                                            initialize=convert_to_dictionary(get_characteristic(data_evs, 'Target (%)')/100),
                                            doc='Vehicle schedule required')
    model.v2gChCost = 1
    model.v2gDchCost = 1

    # @TODO CS -> There is no information from CSs on Manuel's excel.

# variables:
def vars(model):
    model.imports = pe.Var(model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Imported power')
    model.exports = pe.Var(model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Exported power')
    
    model.grid = pe.Var(model.t, within=pe.Reals, initialize=0,
                        doc='Power Grid')
    model.Xgrid = pe.Var(model.t, within=pe.Binary, initialize=1,
                        doc='Power Grid Definition')

    model.P_import_relax = pe.Var(model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Power Grid Import Relax Definition')
    model.genActPower = pe.Var(model.gen, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Active power generation')
    model.genExcPower = pe.Var(model.gen, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Excess power generation')
    model.genXo = pe.Var(model.gen, model.t, within=pe.Binary, initialize=0,
                        doc='Generation on/off')
    
    model.loadXo = pe.Var(model.loads, model.t, within=pe.Binary, initialize=0,
                        doc='Load on/off')
    
    model.storState = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='State of charge')
    model.storCharge = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Charging power')
    model.storDischarge = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Discharging power')
    model.storRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxation variable')
    model.storChXo = pe.Var(model.stor, model.t, within=pe.Binary, initialize=0,
                            doc='Charging on/off')
    model.storDchXo = pe.Var(model.stor, model.t, within=pe.Binary, initialize=0,
                            doc='Discharging on/off')
    model.storScheduleChRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxtion variable for following schedule of charging')
    model.storScheduleDchRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxtion variable for following schedule of discharging')


    model.v2gCharge = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Charging power')
    model.v2gDischarge = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Discharging power')
    model.v2gState = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='State of charge')
    model.v2gRelax = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxation variable')
    model.v2gChXo = pe.Var(model.v2g, model.t, within=pe.Binary, initialize=0,
                        doc='Charging on/off')
    model.v2gDchXo = pe.Var(model.v2g, model.t, within=pe.Binary, initialize=0,
                            doc='Discharging on/off')
    model.v2gScheduleChRelax = pe.Var(model.v2g, model.t, within=pe.Reals, initialize=0,
                            doc='Relaxtion variable for following schedule of charging')
    model.v2gScheduleDchRelax = pe.Var(model.v2g, model.t, within=pe.Reals, initialize=0,
                            doc='Relaxtion variable for following schedule of discharging')

# Constraints:

def constraints(model):
    # Limits for the system, grid import power
    def _impMaxEq(m, t):
        return m.imports[t] <= m.impMax * m.Xgrid[t]

    model.impMaxEq = pe.Constraint(model.t, rule=_impMaxEq,
                                doc='Maximum import power')

    # Limits for the system, grid import power contracted
    def _impMaxEq_contracted(m, t):
        return m.imports[t] <= m.impMax + m.P_import_relax[t]

    model.impMaxEq_contracted_cons = pe.Constraint(model.t, rule=_impMaxEq_contracted,
                                doc='Maximum import power')


    def _expMaxEq(m, t):
        return m.exports[t] <= m.expMax * (1 - m.Xgrid[t])

    model.expMaxEq = pe.Constraint(model.t, rule=_expMaxEq,
                                doc='Maximum export power')

    # Upper limit for the PV generator, both types are PV generators, but, type 2 can be curtailed 
    def _genActMaxEq(m, g, t):
        if m.genType[g] == 1:
            return m.genActPower[g, t] <= m.genMax[g]
        elif m.genType[g] == 2:
            return m.genActPower[g, t] + m.genExcPower[g, t] == m.genMax[g]
        return default_behaviour

    model.genActMaxEq = pe.Constraint(model.gen, model.t, rule=_genActMaxEq,
                                        doc='Maximum active power generation')

    #@TODO NÃO TEM LIMITE INFERIOR PARA OS GERADORES DO MODELO DO MANUEL
    # Lower limit for the PV generator, type 1 controllable
    #def _genActMinEq(m, g, t):
    #    if m.genType[g] == 1:
    #        return m.genActPower[g, t] >= m.genMin[g] * m.genXo[g, t]
    #    return default_behaviour

    #model.genActMinEq = pe.Constraint(model.gen, model.t, rule=_genActMinEq,
    #                                doc='Minimum active power generation')

    # Battery discharge limit 
    def _storDchRateEq(m, s, t):
        return m.storDischarge[s, t] <= m.storDchMax[s] * m.storDchXo[s, t]
    #model.storDchRateEq = pe.Constraint(model.stor, model.t, rule=_storDchRateEq,
    #                                        doc='Maximum discharging rate')
    # Battery power charge limit
    def _storChRateEq(m, s, t):
        return m.storCharge[s, t] <= m.storChMax[s] * m.storChXo[s, t]
    #model.storChRateEq = pe.Constraint(model.stor, model.t, rule=_storChRateEq,
    #                                    doc='Maximum charging rate')
    # Battery energy limit 
    def _storMaxEq(m, s, t):
        return m.storState[s, t] <= m.storMax[s]
    #model.storMaxEq = pe.Constraint(model.stor, model.t, rule=_storMaxEq,
    #                                    doc='Maximum energy capacity')
    # Battery energy limit considering the relax variable  
    def _storRelaxEq(m, s, t):
        return m.storState[s, t] >= m.storMax[s] * m.storMin[s]  - m.storRelax[s, t]
    #model.storRelaxEq = pe.Constraint(model.stor, model.t, rule=_storRelaxEq,
    #                                    doc='Relaxation variable')
    # Energy balance in the battery
    def _storBalanceEq(m, s, t):
        if t == m.t.first():
            return m.storState[s, t] == m.storMax[s] * m.storStart[s] + m.storCharge[s, t] * m.storChEff[s] * 24/m.t.last() - m.storDischarge[s, t] * 24/m.t.last() / m.storDchEff[s]
        elif t > m.t.first():
            return m.storState[s, t] == m.storState[s, t - 1] + m.storCharge[s, t] * m.storChEff[s] * 24/m.t.last() - m.storDischarge[s, t] * 24/m.t.last() / m.storDchEff[s]
        return default_behaviour
    #model.storBalanceEq = pe.Constraint(model.stor, model.t, rule=_storBalanceEq,
    #                                        doc='Energy balance')
    # Binary limits for the battery
    def _storChDchEq(m, s, t):
        return m.storChXo[s, t] + m.storDchXo[s, t] <= 1
    #model.storChDchEq = pe.Constraint(model.stor, model.t, rule=_storChDchEq,
    #                                    doc='Charging and discharging are mutually exclusive')

    def storMinExitSoC(m, s, t):
        if t == m.t.last():
            return m.storState[s, t] >= m.storBatteryTarget[s] * m.storMax[s]
        return default_behaviour
    #model.storMinExitSoC_cons = pe.Constraint(model.stor, model.t, rule=storMinExitSoC,
    #                                    doc='Min Exit Soc')

    # EV power discharge limit
    def _v2gDchRateEq(m, v, t):
        return m.v2gDischarge[v, t] <= m.v2gDchMax[v] * m.v2gDchXo[v, t]
    #model.v2gDchRateEq = pe.Constraint(model.v2g, model.t, rule=_v2gDchRateEq,
    #                                    doc='Maximum discharging rate')
    # EV power charge limit
    def _v2gChRateEq(m, v, t):
        return m.v2gCharge[v, t] <= m.v2gChMax[v] * m.v2gChXo[v, t]
    #model.v2gChRateEq = pe.Constraint(model.v2g, model.t, rule=_v2gChRateEq,
    #                                    doc='Maximum charging rate')
    # Energy in the battery limit
    def _v2gMaxEq(m, v, t):
        return m.v2gState[v, t] <= m.v2gMax[v]
    #model.v2gMaxEq = pe.Constraint(model.v2g, model.t, rule=_v2gMaxEq,
    #                                doc='Maximum energy capacity')
    # To validate the SOC requiered when the EV connects to the CS
    def _v2gRelaxEq(m, v, t):
    # First it is validated if the EV is connected to the CS with this if
        if m.v2gConnected[v, t] == 1: #Quando chega ao ponto de carregamento tem que estar com maior SOC que o minimo
        #Then, it is validated the SOC required in energy, in case of this SOC being 0, the energy in the EV battery must be atleast the minimum 
            if m.v2gScheduleDepartureSOC[v] == 0:
                return m.v2gState[v, t] >= m.v2gMin[v] - m.v2gRelax[v, t]
            # In case of the SOC required will be differente from 0, the energy in the battery must be more than this soc 
            else:
                return m.v2gState[v, t] >= m.v2gScheduleDepartureSOC[v] - m.v2gRelax[v, t]
        else:
            return default_behaviour
        
    #model.v2gRelaxEq = pe.Constraint(model.v2g, model.t, rule=_v2gRelaxEq,
    #                                    doc='Relaxation variable')
    # Energy balance in the EV battery    
    def _v2gStateEq(m, v, t):
        if m.v2gConnected[v, t] == 0: # If vehicle is not scheduled (not connected to the CS)
            return m.v2gState[v, t] == 0
        elif (m.v2gConnected[v, t] == 1) & (t == m.t.first()): # If vehicle is scheduled and it is the first time step
            return m.v2gState[v, t] == m.v2gScheduleArrivalSOC[v] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
        elif t > 1: # If not the first time step
            if (m.v2gConnected[v, t - 1] == 1) & (m.v2gConnected[v, t] == 1): # If was and is currently connected
                return m.v2gState[v, t] == m.v2gState[v, t - 1] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
            elif (m.v2gConnected[v, t - 1] == 0) & (m.v2gConnected[v, t] == 1): # If became connected
                return m.v2gState[v, t] == m.v2gScheduleArrivalSOC[v] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
        return default_behaviour

    #model.v2gStateEq = pe.Constraint(model.v2g, model.t, rule=_v2gStateEq,
    #                                    doc='State of charge')
    # Binary limit for the EV battery charging and discharging process
    def _v2gChDchEq(m, v, t):
        #return m.v2gCharge[v, t] + m.v2gDischarge[v, t] <= 1
        return m.v2gChXo[v, t] + m.v2gDchXo[v, t] <= 1
    #model.v2gChDchEq = pe.Constraint(model.v2g, model.t, rule=_v2gChDchEq,
    #                                    doc='Charging and discharging cannot occur simultaneously')

    def _balanceEq(m, t):
        temp_gens = sum([m.genActPower[g, t] - m.genExcPower[g, t]
                        for g in np.arange(1, m.gen.last() + 1)])

        #temp_stor = sum([m.storCharge[s, t] - m.storDischarge[s, t]
        #                for s in np.arange(1, m.stor.last() + 1)])

        #temp_v2g = sum([m.v2gCharge[v, t] - m.v2gDischarge[v, t]
        #                for v in np.arange(1, m.v2g.last() + 1)])
        
        temp_load = sum([m.loadValues[l, t] for l in np.arange(1, m.loads.last() + 1)])
        #return temp_gens - temp_stor - temp_v2g + temp_load + m.imports[t] - m.exports[t] == 0
        return temp_gens + temp_load + m.imports[t] - m.exports[t] == 0
    
    model.balanceEq = pe.Constraint(model.t, rule=_balanceEq,
                                        doc='Balance equation')

    #@TODO Não está a ser utilizada esta obj function
    def _objFn(m):
        temp_gens = sum([m.genActPower[g, t] * m.genActCost + m.genExcPower[g, t] * m.genExcCost
                        for t in np.arange(m.t.first(), m.t.last() + 1) for g in np.arange(1, m.gen.last() + 1)])

        temp_stor = sum([m.storDischarge[s, t] * m.storDchCost +\
                        m.storCharge[s, t] * m.storChCost +\
                        (m.storScheduleChRelax[s, t])**2 * 2e-3 +\
                        (m.storScheduleDchRelax[s, t])**2 * 2e-3 +\
                        m.storRelax[s, t] *  1
                        for t in np.arange(m.t.first(), m.t.last() + 1) for s in np.arange(1, m.stor.last() + 1)])
        
        temp_v2g = sum([m.v2gCharge[v, t] * m.v2gChCost + \
                        m.v2gDischarge[v, t] * m.v2gDchCost +\
                        (m.v2gScheduleChRelax[v, t])**2 * 0.01 +\
                        (m.v2gScheduleDchRelax[v, t])**2 * 0.01 +\
                        m.v2gRelax[v, t] * 200
                        for t in np.arange(m.t.first(), m.t.last() + 1) for v in np.arange(1, m.v2g.last() + 1)])
        
        temp_rest = sum([m.imports[t] * m.impCost - m.exports[t] * m.expCost + m.P_import_relax[t] * 100
                        for t in np.arange(m.t.first(), m.t.last() + 1)])

        return temp_gens + temp_stor + temp_v2g + temp_rest


    def obj_rule(model):
        return sum(model.imports[t] for t in np.arange(model.t.first(), model.t.last() + 1))

    model.obj = pe.Objective(expr=obj_rule, sense=pe.minimize)
