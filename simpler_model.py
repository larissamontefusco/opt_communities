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
    
    # LOADS:
    model.loads = pe.Set(initialize=np.arange(1, len(get_characteristic(data_load, 'Nominal Power (kW)')) + 1),doc='Number of loads')
    
    model.loadValues = pe.Param(model.loads, model.t, initialize=convert_to_dictionary(get_timeseries(data_load, 'Consumption (kW)')),
                                doc='Load Power During time')
    
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

def constraints(model):
    # Upper limit for the PV generator, both types are PV generators, but, type 2 can be curtailed 
    def _genActMaxEq(m, g, t):
        if m.genType[g] == 1:
            return m.genActPower[g, t] <= m.genMax[g]
        elif m.genType[g] == 2:
            return m.genActPower[g, t] + m.genExcPower[g, t] == m.genMax[g]
        return default_behaviour

    model.genActMaxEq = pe.Constraint(model.gen, model.t, rule=_genActMaxEq,
                                        doc='Maximum active power generation')

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
    
    # Limits for the system, grid export power contracted
    def _expMaxEq(m, t):
        return m.exports[t] <= m.expMax * (1 - m.Xgrid[t])

    model.expMaxEq = pe.Constraint(model.t, rule=_expMaxEq,
                                doc='Maximum export power')
    
    def _balanceEq(m, t):
        temp_gens = sum([m.genActPower[g, t] - m.genExcPower[g, t]
                        for g in np.arange(1, m.gen.last() + 1)])
        temp_load = sum([m.loadValues[l, t] for l in np.arange(1, m.loads.last() + 1)])
        #return temp_gens - temp_stor - temp_v2g + temp_load + m.imports[t] - m.exports[t] == 0
        return temp_gens + temp_load + m.imports[t] - m.exports[t] == 0
    
    model.balanceEq = pe.Constraint(model.t, rule=_balanceEq,
                                        doc='Balance equation')
    def obj_rule(model):
        return sum(model.genActPower[g, t] for t in np.arange(model.t.first(), model.t.last() + 1)
                   for g in np.arange(1, model.gen.last() + 1) if model.genType[g] == 1)
    
    model.obj = pe.Objective(expr=obj_rule, sense=pe.minimize)