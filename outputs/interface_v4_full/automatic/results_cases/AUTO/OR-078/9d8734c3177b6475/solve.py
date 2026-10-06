import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
if set(options) != set(cost_per_lot.keys()) or set(options) != set(gen_per_lot.keys()):
    raise ValueError('Mismatch in option keys between index set and parameter dictionaries.')
total_demand = 200.0
m = gp.Model('Electricity_Procurement_Lot_Selection')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.optimize()