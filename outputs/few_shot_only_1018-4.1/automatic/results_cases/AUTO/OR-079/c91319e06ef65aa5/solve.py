import pandas as pd
import gurobipy as gp
from gurobipy import GRB
facility_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv', dtype=str, keep_default_na=False, delim_whitespace=True)
facility_ids = facility_df['Facility'].tolist()
fixed_cost = {row['Facility']: float(row['FixedCost']) for _, row in facility_df.iterrows()}
capacity = {row['Facility']: float(row['Capacity']) for _, row in facility_df.iterrows()}
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv', dtype=str, keep_default_na=False, delim_whitespace=True)
demand_ids = demand_df['Destination'].tolist()
demand = {row['Destination']: float(row['Demand']) for _, row in demand_df.iterrows()}
shipping_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv', dtype=str, keep_default_na=False, delim_whitespace=True)
shipping_df.set_index('Origin', inplace=True)
shipping_cost = {i: {j: float(shipping_df.loc[i, j]) for j in demand_ids} for i in facility_ids}
if set(fixed_cost.keys()) != set(facility_ids) or set(capacity.keys()) != set(facility_ids):
    raise ValueError('Mismatch in facility identifiers between cost/capacity and facility list.')
if set(demand.keys()) != set(demand_ids):
    raise ValueError('Mismatch in demand identifiers between demand and demand list.')
if set(shipping_cost.keys()) != set(facility_ids):
    raise ValueError('Mismatch in shipping cost facility indices.')
for i in facility_ids:
    if set(shipping_cost[i].keys()) != set(demand_ids):
        raise ValueError(f'Mismatch in shipping cost demand indices for facility {i}.')
m = gp.Model('ElectroTech_Network_Design')
y_vars = m.addVars(facility_ids, vtype=GRB.BINARY, name='')
x_vars = m.addVars(facility_ids, demand_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in facility_ids)) + gp.quicksum((shipping_cost[i][j] * x_vars[i, j] for i in facility_ids for j in demand_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in facility_ids)) == demand[j] for j in demand_ids), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in demand_ids)) <= capacity[i] * y_vars[i] for i in facility_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')