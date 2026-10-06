import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
if energy_df['cost_per_lot'].isnull().any():
    raise ValueError("Missing values found in 'cost_per_lot' column.")
if energy_df['gen_per_lot'].isnull().any():
    raise ValueError("Missing values found in 'gen_per_lot' column.")
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
total_demand = 200.0
m = gp.Model('Electricity_Procurement_Lot_Sizing')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='DemandSatisfaction')
m.optimize()