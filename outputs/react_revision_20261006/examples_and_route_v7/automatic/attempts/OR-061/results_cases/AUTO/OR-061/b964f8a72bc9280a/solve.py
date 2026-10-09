import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', dtype=str, keep_default_na=False)
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
branches = demand_df['customer'].tolist()
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    supplier = row['Unnamed: 0']
    try:
        fixed_costs[supplier] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed_costs value for supplier {supplier}')
demand = {}
for (idx, row) in demand_df.iterrows():
    branch = row['customer']
    try:
        demand[branch] = int(row['demand'])
    except Exception:
        raise ValueError(f'Invalid demand value for branch {branch}')
transport_costs = {}
for (idx, row) in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    for branch in branches:
        try:
            cost = float(row[branch])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, branch {branch}')
        transport_costs[supplier, branch] = cost
if set(suppliers) != set(transport_df['Unnamed: 0']):
    raise ValueError('Mismatch in supplier identifiers between fixed_cost.csv and transportation_costs.csv')
if set(branches) != set(transport_df.columns[1:]):
    raise ValueError('Mismatch in branch identifiers between demand.csv and transportation_costs.csv')
m = gp.Model('UFLP_Superstore')
x_keys = [(i, j) for i in suppliers for j in branches]
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
obj = gp.quicksum((fixed_costs[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_costs[i, j] * x_vars[i, j] for i in suppliers for j in branches))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in branches:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')