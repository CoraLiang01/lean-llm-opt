import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
option_ids = energy_df['option'].astype(str).tolist()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
tech = energy_df.set_index('option')['tech'].astype(str).to_dict()
for oid in option_ids:
    if oid not in cost_per_lot or oid not in gen_per_lot:
        raise ValueError(f"Missing cost_per_lot or gen_per_lot for option '{oid}'.")
total_demand = 200.0

def solve_generation_lot_problem():
    m = gp.Model('Electricity_Lot_Purchasing')
    x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[oid] * x[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[oid] * x[oid] for oid in option_ids)) >= total_demand, name='demand')
    m.optimize()
    return m
m = solve_generation_lot_problem()