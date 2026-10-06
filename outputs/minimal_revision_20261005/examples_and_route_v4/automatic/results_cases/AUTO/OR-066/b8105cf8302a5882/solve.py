import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
if demand_df['customer'].isnull().any() or demand_df['demand'].isnull().any():
    raise ValueError('Missing values in demand.csv')
customers = demand_df['customer'].astype(str).tolist()
demand = dict(zip(demand_df['customer'].astype(str), demand_df['demand'].astype(float)))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
if fixed_cost_df['Unnamed: 0'].isnull().any() or fixed_cost_df['fixed_costs'].isnull().any():
    raise ValueError('Missing values in fixed_cost.csv')
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str), fixed_cost_df['fixed_costs'].astype(float)))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
if trans_cost_df['Unnamed: 0'].isnull().any():
    raise ValueError('Missing supplier IDs in transportation_costs.csv')
trans_suppliers = trans_cost_df['Unnamed: 0'].astype(str).tolist()
if set(suppliers) != set(trans_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(trans_suppliers)}')
trans_customers = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(trans_customers):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_customers)}')
transport_costs = {}
for (_, row) in trans_cost_df.iterrows():
    i = str(row['Unnamed: 0'])
    for j in customers:
        val = row[j]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
        transport_costs[i, j] = float(val)
I = suppliers
J = customers
M = sum((demand[j] for j in J))

def solve_uflp(I, J, demand, fixed_costs, transport_costs, M):
    m = gp.Model('UFLP')
    x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((fixed_costs[i] * y[i] for i in I)) + gp.quicksum((transport_costs[i, j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_uflp(I, J, demand, fixed_costs, transport_costs, M)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')