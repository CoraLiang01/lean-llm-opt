import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
cost_df['plant'] = cost_df['plant'].astype(str).str.strip()
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
plants = [f'F{i}' for i in range(1, 16)]
customers = [f'C{i}' for i in range(1, 16)]
missing_plants = set(plants) - set(cost_df['plant'])
if missing_plants:
    raise ValueError(f'Missing plant(s) in cost.csv: {missing_plants}')
missing_customers = set(customers) - set(demand_df['customer'])
if missing_customers:
    raise ValueError(f'Missing customer(s) in demand.csv: {missing_customers}')
for c in customers:
    if c not in cost_df.columns:
        raise ValueError(f'Missing transport cost column for customer {c} in cost.csv')
fixed_cost = cost_df.set_index('plant')['fixed_cost'].astype(float).to_dict()
capacity = cost_df.set_index('plant')['capacity'].astype(float).to_dict()
transport_cost = {}
for i in plants:
    row = cost_df[cost_df['plant'] == i].iloc[0]
    for j in customers:
        transport_cost[i, j] = float(row[j])
demand = demand_df.set_index('customer')['demand'].astype(float).to_dict()
m = gp.Model('UFLP')
m.Params.MIPGap = 0.0001
x = m.addVars(plants, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')