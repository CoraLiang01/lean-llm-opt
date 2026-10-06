import pandas as pd
import gurobipy as gp
from gurobipy import GRB
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', sep=',')
options = energy_df['option'].astype(str).tolist()
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot']))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot']))
for i in options:
    if i not in gen_per_lot or i not in cost_per_lot:
        raise ValueError(f'Missing parameter data for option {i}')
total_demand = 200
m = gp.Model('electricity_lot_sizing')
x = m.addVars(options, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.Status}')