import pandas as pd
import gurobipy as gp
from gurobipy import GRB
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', sep=',')
options = energy_df['option'].astype(str).tolist()
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot']))
demand = 200

def solve_problem():
    m = gp.Model('electricity_lot_sizing')
    x = m.addVars(options, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= demand, name='demand')
    m.optimize()
    return m
m = solve_problem()