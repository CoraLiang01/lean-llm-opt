import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['Unnamed: 0'] = df_supply['Unnamed: 0'].astype(str).str.strip()
sources = df_supply['Unnamed: 0'].tolist()
supply_capacity = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_columns = [col for col in df_cost.columns if col.startswith('C')]
if set(cost_columns) != set(customers):
    raise ValueError(f'Customer columns in transportation_costs.csv do not match customer_demand.csv: {cost_columns} vs {customers}')
if set(df_cost['Unnamed: 0']) != set(sources):
    raise ValueError(f"Source rows in transportation_costs.csv do not match supply_capacity.csv: {list(df_cost['Unnamed: 0'])} vs {sources}")
cost = {}
for (_, row) in df_cost.iterrows():
    s = str(row['Unnamed: 0']).strip()
    for c in customers:
        cost[s, c] = float(row[c])
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list.')
if set(supply_capacity.keys()) != set(sources):
    raise ValueError('Mismatch in supply capacity keys and source list.')
for s in sources:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing cost entry for source {s}, customer {c}.')
m = gp.Model('TransportationOptimization')
x = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
for s in sources:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan (amounts shipped from each source to each customer) ---')
    for s in sources:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  From {s} to {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')