import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
customer_demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['store'] = df_supply['Unnamed: 0'].astype(str).str.strip()
stores = df_supply['store'].tolist()
supply_capacity = dict(zip(df_supply['store'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['store'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [c for c in df_cost.columns if c.startswith('C')]
missing_customers = set(customers) - set(cost_customer_cols)
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
missing_stores = set(stores) - set(df_cost['store'])
if missing_stores:
    raise ValueError(f'Missing stores in transportation_costs.csv: {missing_stores}')
cost = {}
for (_, row) in df_cost.iterrows():
    s = row['store']
    for c in customers:
        cost[s, c] = float(row[c])
for s in stores:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing cost coefficient for store {s}, customer {c}')
for s in stores:
    if s not in supply_capacity:
        raise ValueError(f'Missing supply capacity for store {s}')
for c in customers:
    if c not in customer_demand:
        raise ValueError(f'Missing demand for customer {c}')
index_pairs = [(s, c) for s in stores for c in customers]

def solve_problem():
    m = gp.Model('Walmart_Transportation')
    x = m.addVars(index_pairs, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for (s, c) in index_pairs)), gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[s, c] for s in stores)) == customer_demand[c], name=f'demand_{c}')
    for s in stores:
        m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')