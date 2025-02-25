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
from pprint import pprint


class DC_Community_Opt():
    def __init__(self, excel_file):
        self.default_behaviour = pe.Constraint.Skip

        # READING EXCEL:
        self.data_edf_pu_profile = pd.read_excel(excel_file, sheet_name="User-defined Assets Profiles", header=None)
        self.data_edf_pu_profile.columns = self.data_edf_pu_profile.iloc[0]
        self.data_edf_assets = pd.read_excel(excel_file, sheet_name="Assets Nodes", header=None)
        self.data_edf_assets.columns = self.data_edf_assets.iloc[0]
        self.data_edf_geral = pd.read_excel(excel_file, sheet_name="UC Definition", header=None)

        self.edf_components = {}
        self.edf_in_use_components = {}
        self.model = pe.ConcreteModel()
        
        
    # Auxiliary function to convert numpy arrays to dictionaries
    def convert_to_dictionary(self, a, t_start=0):
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

    def get_row_timeseries(self, values, component: str):

        temp_data_idx = np.where(values == component)
        temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
        for i in np.arange(temp_data.shape[0]):
            temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
        temp_data = temp_data[:, temp_data_idx[1][0] + 1:]

        return temp_data

    def get_column_timeseries(self, values, component: str):
        temp_data_idx = np.where(values == component)  # Localiza a posição do componente
        if len(temp_data_idx[0]) == 0:
            return None  # Retorna None se o componente não for encontrado
        
        row_idx, col_idx = temp_data_idx[0][0], temp_data_idx[1][0]  # Primeira ocorrência
        
        # Pega todas as linhas abaixo da linha onde `component` foi encontrado, na mesma coluna
        temp_data = values.iloc[row_idx + 1:, col_idx].copy(deep=True)
        
        # Converte os valores para numérico, substituindo erros por NaN
        #temp_data = pd.to_numeric(temp_data, errors='coerce')
        temp_data = temp_data.dropna().to_numpy()

        return temp_data

    def get_characteristic(self, values, component: str, keep_string: bool = False):

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
    
    
    def extract_excel_data(self):
        self.edf_components['component_type'] = self.get_column_timeseries(self.data_edf_assets, "Component type")
        self.edf_components['nominal_power'] = self.get_column_timeseries(self.data_edf_assets, "Maximum power (kW)")
        self.edf_components['node_number'] = self.get_column_timeseries(self.data_edf_assets, "Node number")
        self.edf_components['max_capacity'] = self.get_column_timeseries(self.data_edf_assets, 'Capacity (kWh)')

        node_number = self.get_row_timeseries(self.data_edf_pu_profile, '               Node\nTime step')
        self.edf_in_use_components['node_number'] = np.array(node_number.flatten(), dtype=object)
        self.edf_in_use_components['nodes_types'] = list(set([
            (self.edf_components['component_type'][i], value)
            for i, value in enumerate(self.edf_components['node_number'])
            for j, another_value in enumerate(self.edf_in_use_components['node_number'])
            if value == another_value or self.edf_components['component_type'][i] == 'AC Grid'
        ]))
                    
        # AC GRID:
        self.edf_in_use_components['ac_grid_ids'] = [t[1] for t in self.edf_in_use_components['nodes_types'] if t[0] == "AC Grid"]
        self.edf_in_use_components['n_ac_grid'] = len(self.edf_in_use_components['ac_grid_ids'])
        self.edf_in_use_components['ac_grid_nominal_power'] = [
            self.edf_components['nominal_power'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['ac_grid_ids']
        ]

        # EVS:
        self.edf_in_use_components['evs_ids'] = [t[1] for t in self.edf_in_use_components['nodes_types'] if t[0] == "EV"]
        self.edf_in_use_components['n_evs'] = len(self.edf_in_use_components['evs_ids'])

        self.edf_in_use_components['evs_pu_profile'] = [list(self.get_column_timeseries(self.data_edf_pu_profile, t[1])) 
                                                        for t in self.edf_in_use_components['nodes_types'] 
                                                        if t[0] == "EV"]

        self.edf_in_use_components['ev_nominal_power'] = [
            self.edf_components['nominal_power'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['evs_ids']
        ]

        self.edf_in_use_components['ev_power_profile'] = [
            [pu * power for pu in profile] 
            for profile, power in zip(self.edf_in_use_components['evs_pu_profile'], 
                                    self.edf_in_use_components['ev_nominal_power'])
        ]
        self.edf_in_use_components['ev_max_capacity'] = [
            self.edf_components['max_capacity'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['evs_ids']
        ]

        # BESS:

        self.edf_in_use_components['bess_ids'] = [t[1] for t in self.edf_in_use_components['nodes_types'] if t[0] == "Storage"]
        self.edf_in_use_components['n_storage'] = len(self.edf_in_use_components['bess_ids'])

        self.edf_in_use_components['bess_nominal_power'] = [
            self.edf_components['nominal_power'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['bess_ids']
        ]
        self.edf_in_use_components['bess_pu_profile'] = [list(self.get_column_timeseries(self.data_edf_pu_profile, t[1])) 
                                                         for t in self.edf_in_use_components['nodes_types'] 
                                                         if t[0] == "Storage"]

        self.edf_in_use_components['bess_power_profile'] = [
            [pu * power for pu in profile] 
            for profile, power in zip(self.edf_in_use_components['bess_pu_profile'], 
                                    self.edf_in_use_components['bess_nominal_power'])
        ]
        self.edf_in_use_components['bess_max_capacity'] = [
            self.edf_components['max_capacity'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['bess_ids']
        ]

        # PV:
        self.edf_in_use_components['pv_ids'] = [t[1] for t in self.edf_in_use_components['nodes_types'] if t[0] == "PV "]
        self.edf_in_use_components['n_pvs'] = len(self.edf_in_use_components['pv_ids'])

        self.edf_in_use_components['pv_pu_profile'] = [list(self.get_column_timeseries(self.data_edf_pu_profile, t[1])) 
                                                       for t in self.edf_in_use_components['nodes_types'] 
                                                       if t[0] == "PV "]

        self.edf_in_use_components['pv_nominal_power'] = [
            edf_components['nominal_power'][i]
            for i, value in enumerate(edf_components['node_number'])
            if value in self.edf_in_use_components['pv_ids']
        ]
        self.edf_in_use_components['pv_power_profile'] = [
            [pu * power for pu in profile] 
            for profile, power in zip(self.edf_in_use_components['pv_pu_profile'], 
                                    self.edf_in_use_components['pv_nominal_power'])
        ]

        # LOADS:

        self.edf_in_use_components['load_ids'] = [t[1] for t in self.edf_in_use_components['nodes_types'] if t[0] == "DC Load"]
        self.edf_in_use_components['n_loads'] = len(self.edf_in_use_components['load_ids'])

        self.edf_in_use_components['load_pu_profile'] = [list(self.get_column_timeseries(self.data_edf_pu_profile, t[1])) 
                                                         for t in self.edf_in_use_components['nodes_types'] 
                                                         if t[0] == "DC Load"]

        self.edf_in_use_components['load_nominal_power'] = [
            self.edf_components['nominal_power'][i]
            for i, value in enumerate(self.edf_components['node_number'])
            if value in self.edf_in_use_components['load_ids']
        ]

        self.edf_in_use_components['load_power_profile'] = [
            [pu * power for pu in profile] 
            for profile, power in zip(self.edf_in_use_components['load_pu_profile'], 
                                    self.edf_in_use_components['load_nominal_power'])
        ]

        self.edf_in_use_components['impMax'] = self.get_characteristic(self.data_edf_geral, 'Total PV power installed (kWp)')               
        self.edf_in_use_components['expMax'] = self.get_characteristic(self.data_edf_geral, 'Total PV power installed (kWp)')
        
        self.edf_in_use_components['period'] = self.get_characteristic(self.data_edf_geral, 'Simulation period (days)')
        self.edf_in_use_components['timestep'] = self.get_characteristic(self.data_edf_geral, 'Simulation time step (mins)')
        self.edf_in_use_components['dT'] =  self.edf_in_use_components['timestep'] / 60
        self.edf_in_use_components['n_time'] = self.edf_in_use_components['period']*24*self.edf_in_use_components['dt']

    def params(self):
        self.model.t = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_time'] + 1),
                    doc='Time periods')
        self.model.impMax = self.edf_in_use_components['impMax']
        self.model.expMax = self.edf_in_use_components['expMax']
        
        # GENERATORS:
        self.model.gen = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_pvs']),
                            doc='Number of generators')
        self.model.genMax = pe.Param(self.model.gen, initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['pv_nominal_power'])),
                                    doc='Gen Max')
        
        self.model.genValues = pe.Param(self.model.gen, self.model.t,
                                initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['pv_power_profile'])),
                                doc='Forecasted power generation')
        
        # AC GENERATORS:
        self.model.ac_gen = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_ac_grid']),
                            doc='Number of generators')
        self.model.acGenMax = pe.Param(self.model.ac_gen, initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['ac_grid_nominal_power'])),
                                    doc='Gen Max')
        
        # LOADS:
        self.model.loads = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_loads'] + 1), 
                            doc='Number of loads')
        
        self.model.loadValues = pe.Param(self.model.loads, self.model.t, 
                                         initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['load_power_profile'])),
                                         doc='Load Power During time')
        
        # BESS:
        self.model.stor = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_bess'] + 1), 
                                 doc='Number of storage units')

        self.model.storMax = pe.Param(self.model.stor, initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['bess_max_capacity']))) #kW

        self.model.storDchMax = pe.Param(self.model.stor,
                                    initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['bess_nominal_power'])),
                                    doc='Starting energy capacity')
        self.model.storChMax = pe.Param(self.model.stor,
                                    initialize=self.convert_to_dictionary(np.array(self.edf_in_use_components['bess_nominal_power'])),
                                    doc='Starting energy capacity')
    
        self.model.storMin = pe.Param(self.model.stor,
                            initialize=self.convert_to_dictionary(),
                            doc='Starting energy capacity')

        self.model.storStart = pe.Param(self.model.stor,
                                initialize=convert_to_dictionary())

        self.model.storDchEff = pe.Param(self.model.stor,
                                initialize=convert_to_dictionary(),
                                doc='Starting SoC (%)')
        self.model.storChEff = pe.Param(self.model.stor,
                                initialize=convert_to_dictionary(),
                                doc='Starting SoC (%)')
        self.model.storBatteryTarget = pe.Param(self.model.stor,
                                initialize=convert_to_dictionary(),
                                doc='Target SoC (%)')
        
        # EVS:
        self.model.v2g = pe.Set(initialize=np.arange(1, self.edf_in_use_components['n_evs'] + 1),
                                doc='Number of EVs')
        
        self.model.v2gDchMax = pe.Param(self.model.v2g, 
                                initialize=self.convert_to_dictionary(),
                                doc='Maximum scheduled discharging power')
        self.model.v2gChMax = pe.Param(self.model.v2g, 
                                initialize=self.convert_to_dictionary(),
                                doc='Maximum scheduled charging power')

        self.model.v2gDchEff = pe.Param(self.model.v2g,
                                initialize=self.convert_to_dictionary(),
                                doc='Discharging efficiency')
        self.model.v2gChEff = pe.Param(self.model.v2g,
                                initialize=self.convert_to_dictionary(),
                                doc='Charging efficiency')
        self.model.v2gMax = pe.Param(self.model.v2g,
                                initialize=self.convert_to_dictionary(),
                                doc='Maximum energy capacity')
        self.model.v2gMin = pe.Param(self.model.v2g,
                                initialize=self.convert_to_dictionary(),
                                doc='Minimum energy capacity')
        self.model.v2gConnected = pe.Param(self.model.v2g, self.model.t,
                                    initialize=self.convert_to_dictionary(),
                                    doc='Vehicle schedule')
        self.model.v2gScheduleArrivalEnergy = pe.Param(self.model.v2g,
                                            initialize=self.convert_to_dictionary(),
                                            doc='Vehicle schedule arrival SOC')
        self.model.v2gScheduleTargetEnergy = pe.Param(self.model.v2g,
                                                initialize=self.convert_to_dictionary(),
                                                doc='Vehicle schedule required')
        
    def vars(self):
        # SYSTEM
        self.model.imports = pe.Var(self.model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Imported power')
        self.model.exports = pe.Var(self.model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Exported power')
        
        self.model.grid = pe.Var(self.model.t, within=pe.Reals, initialize=0,
                            doc='Power Grid')
        self.model.Xgrid = pe.Var(self.model.t, within=pe.Binary, initialize=1,
                            doc='Power Grid Definition')

        self.model.P_import_relax = pe.Var(self.model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Power Grid Import Relax Definition')
        # GENERATORS
        self.model.genActPower = pe.Var(self.model.gen, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Active power generation')
        self.model.genExcPower = pe.Var(self.model.gen, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Excess power generation')
        
        self.model.acGenActPower = pe.Var(self.model.ac_gen, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Active power generation AC')
        
        self.model.genXo = pe.Var(self.model.gen, self.model.t, within=pe.Binary, initialize=0,
                            doc='Generation on/off')
        
        # BESS
        self.model.storState = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='State of charge')
        self.model.storCharge = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Charging power')
        self.model.storDischarge = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                    doc='Discharging power')
        self.model.storRelax = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Relaxation variable')
        self.model.storChXo = pe.Var(self.model.stor, self.model.t, within=pe.Binary, initialize=0,
                                doc='Charging on/off')
        self.model.storDchXo = pe.Var(self.model.stor, self.model.t, within=pe.Binary, initialize=0,
                                doc='Discharging on/off')
        self.model.storScheduleChRelax = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Relaxtion variable for following schedule of charging')
        self.model.storScheduleDchRelax = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Relaxtion variable for following schedule of discharging')

        # EVS
        self.model.v2gCharge = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Charging power')
        self.model.v2gDischarge = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                    doc='Discharging power')
        self.model.v2gState = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='State of charge')
        self.model.v2gRelax = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Relaxation variable')
        self.model.v2gChXo = pe.Var(self.model.v2g, self.model.t, within=pe.Binary, initialize=0,
                            doc='Charging on/off')
        self.model.v2gDchXo = pe.Var(self.model.v2g, self.model.t, within=pe.Binary, initialize=0,
                                doc='Discharging on/off')
        self.model.v2gScheduleChRelax = pe.Var(self.model.v2g, self.model.t, within=pe.Reals, initialize=0,
                                doc='Relaxtion variable for following schedule of charging')
        self.model.v2gScheduleDchRelax = pe.Var(self.model.v2g, self.model.t, within=pe.Reals, initialize=0,
                                doc='Relaxtion variable for following schedule of discharging')

    def constraints(self):
        # Upper limit for the PV generator
        def _genActMaxEq(m, g, t):
            return m.genActPower[g, t] + m.genExcPower[g, t] == m.genValues[g, t]
        
        self.model.genActMaxEq = pe.Constraint(self.model.gen, self.model.t, rule=_genActMaxEq,
                                            doc='Maximum active power generation')
        
        # Upper limit for the PV generator, both types are PV generators, but, type 2 can be curtailed 
        def _acGenActMaxEq(m, ac, t):
            return m.acGenActPower[ac, t] <= m.acGenMax[ac]

        self.model.acGenActMaxEq = pe.Constraint(self.model.ac_gen, self.model.t, rule=_acGenActMaxEq,
                                            doc='Maximum active power generation AC')
        # Battery discharge limit 
        def _storDchRateEq(m, s, t):
            return m.storDischarge[s, t] <= m.storDchMax[s] * m.storDchXo[s, t]
        self.model.storDchRateEq = pe.Constraint(self.model.stor, self.model.t, rule=_storDchRateEq,
                                                doc='Maximum discharging rate')
        # Battery power charge limit
        def _storChRateEq(m, s, t):
            return m.storCharge[s, t] <= m.storChMax[s] * m.storChXo[s, t]
        self.model.storChRateEq = pe.Constraint(self.model.stor, self.model.t, rule=_storChRateEq,
                                            doc='Maximum charging rate')
        # Battery energy limit 
        def _storMaxEq(m, s, t):
            return m.storState[s, t] <= m.storMax[s]
        self.model.storMaxEq = pe.Constraint(self.model.stor, self.model.t, rule=_storMaxEq,
                                            doc='Maximum energy capacity')
        # Battery energy limit considering the relax variable  
        def _storRelaxEq(m, s, t):
            return m.storState[s, t] >= m.storMin[s]  - m.storRelax[s, t]
        self.model.storRelaxEq = pe.Constraint(self.model.stor, self.model.t, rule=_storRelaxEq,
                                            doc='Relaxation variable')
        
        # Energy balance in the battery @TODO O PROBLEMA ESTÁ AQUI
        def _storBalanceEq(m, s, t):
            if t == m.t.first():
                return m.storState[s, t] == m.storStart[s] + m.storCharge[s, t] * m.storChEff[s]- m.storDischarge[s, t] / m.storDchEff[s]
            elif t > m.t.first():
                return m.storState[s, t] == m.storState[s, t - 1] + m.storCharge[s, t] * m.storChEff[s] - m.storDischarge[s, t] / m.storDchEff[s]
            return default_behaviour
        self.model.storBalanceEq = pe.Constraint(self.model.stor, self.model.t, rule=_storBalanceEq,
                                                doc='Energy balance')
        
        # Binary limits for the battery
        def _storChDchEq(m, s, t):
            return m.storChXo[s, t] + m.storDchXo[s, t] <= 1
        self.model.storChDchEq = pe.Constraint(self.model.stor, self.model.t, rule=_storChDchEq,
                                            doc='Charging and discharging are mutually exclusive')

        def storMinExitSoC(m, s, t):
            if t == m.t.last():
                return m.storState[s, t] >= m.storBatteryTarget[s] * m.storMax[s]
            return default_behaviour
        self.model.storMinExitSoC_cons = pe.Constraint(self.model.stor, self.model.t, rule=storMinExitSoC,
                                            doc='Min Exit Soc')
        
        # EV power discharge limit
        def _v2gDchRateEq(m, v, t):
            return m.v2gDischarge[v, t] <= m.v2gDchMax[v] * m.v2gDchXo[v, t]
        self.model.v2gDchRateEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gDchRateEq,
                                            doc='Maximum discharging rate')
        # EV power charge limit
        def _v2gChRateEq(m, v, t):
            return m.v2gCharge[v, t] <= m.v2gChMax[v] * m.v2gChXo[v, t]
        self.model.v2gChRateEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gChRateEq,
                                            doc='Maximum charging rate')
        # Energy in the battery limit
        def _v2gMaxEq(m, v, t):
            return m.v2gState[v, t] <= m.v2gMax[v]
        self.model.v2gMaxEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gMaxEq,
                                        doc='Maximum energy capacity')
        # To validate the SOC requiered when the EV connects to the CS
        def _v2gRelaxEq(m, v, t):
        # First it is validated if the EV is connected to the CS with this if
            if m.v2gConnected[v, t] == 1: #Quando chega ao ponto de carregamento tem que estar com maior SOC que o minimo
            #Then, it is validated the SOC required in energy, in case of this SOC being 0, the energy in the EV battery must be atleast the minimum 
                if m.v2gScheduleTargetEnergy[v] == 0:
                    return m.v2gState[v, t] >= m.v2gMin[v] - m.v2gRelax[v, t]
                # In case of the SOC required will be differente from 0, the energy in the battery must be more than this soc 
                else:
                    return m.v2gState[v, t] >= m.v2gScheduleTargetEnergy[v] - m.v2gRelax[v, t]
            else:
                return default_behaviour
            
        self.model.v2gRelaxEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gRelaxEq,
                                            doc='Relaxation variable')
        def _v2gBalanceEq(m, v, t):
            if t == m.t.first():
                return m.v2gState[v, t] == m.v2gScheduleArrivalEnergy[v] + m.v2gCharge[v, t] * m.v2gChEff[v]- m.v2gDischarge[v, t] / m.v2gDchEff[v]
            elif t > m.t.first():
                return m.v2gState[v, t] == m.v2gState[v, t - 1] + m.v2gCharge[v, t] * m.v2gChEff[v] - m.v2gDischarge[v, t] / m.v2gDchEff[v]
            return default_behaviour
        
        self.model.v2gBalanceEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gBalanceEq,
                                            doc='State of charge')
        
        # Binary limit for the EV battery charging and discharging process
        def _v2gChDchEq(m, v, t):
            return m.v2gChXo[v, t] + m.v2gDchXo[v, t] <= 1
        self.model.v2gChDchEq = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gChDchEq,
                                            doc='Charging and discharging cannot occur simultaneously')
        
        def _v2gMinExitSoC(m, v, t):
            if t == m.t.last():
                return m.v2gState[v, t] >= m.v2gScheduleTargetEnergy[v]
            return default_behaviour
        self.model.v2gMinExitSoC = pe.Constraint(self.model.v2g, self.model.t, rule=_v2gMinExitSoC,
                                            doc='Min Exit Soc')
        
            
        def _balanceEq(m, t):
            temp_gens = sum([m.genActPower[g, t] - m.genExcPower[g, t]
                            for g in np.arange(1, m.gen.last() + 1)])
            
            temp_ac_gens = sum(m.acGenActPower[ac, t] for ac in np.arange(1, m.ac_gen.last() + 1))
            
            temp_load = sum([m.loadValues[l, t] for l in np.arange(1, m.loads.last() + 1)])

            temp_stor = sum([m.storCharge[s, t] - m.storDischarge[s, t]
                            for s in np.arange(1, m.stor.last() + 1)])

            temp_v2g = sum([m.v2gCharge[v, t] - m.v2gDischarge[v, t]
                            for v in np.arange(1, m.v2g.last() + 1)])
            
            return temp_gens + temp_ac_gens - temp_load - temp_stor - temp_v2g == 0

        self.model.balanceEq = pe.Constraint(self.model.t, rule=_balanceEq,
                                            doc='Balance equation')
        def obj_rule(m):
            return sum(m.acGenActPower[ac, t] for t in np.arange(m.t.first(), m.t.last() + 1)
                    for ac in np.arange(1, m.ac_gen.last() + 1))
        
        self.model.obj = pe.Objective(expr=obj_rule, sense=pe.minimize)