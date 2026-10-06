import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
if set(cost_per_lot.keys()) != set(options) or set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in parameter keys and option set.')
total_demand = 200.0

def solve_problem():
    m = gp.Model('Electricity_Procurement_MIP')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='Demand')
    m.optimize()
    return m
m = solve_problem()