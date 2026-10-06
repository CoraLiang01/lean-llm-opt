import pandas as pd
import numpy as np
from gurobipy import Model, GRB
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', sep=',')
energy_df['option'] = energy_df['option'].astype(str)
energy_df['tech'] = energy_df['tech'].astype(str)
energy_df['gen_per_lot'] = energy_df['gen_per_lot'].astype(float)
energy_df['cost_per_lot'] = energy_df['cost_per_lot'].astype(float)
options = energy_df['option'].tolist()
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
tech = dict(zip(energy_df['option'], energy_df['tech']))
demand = 200.0
if set(cost_per_lot.keys()) != set(options) or set(gen_per_lot.keys()) != set(options):
    raise ValueError('Missing cost or generation coefficients for some options.')
m = Model('electricity_lot_sizing')
x = m.addVars(options, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((cost_per_lot[i] * x[i] for i in options)), GRB.MINIMIZE)
m.addConstr(sum((gen_per_lot[i] * x[i] for i in options)) >= demand, name='demand')
m.optimize()