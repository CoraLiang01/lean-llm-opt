import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
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
cost_sources = df_cost['Unnamed: 0'].tolist()
cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
if set(sources) != set(cost_sources):
    raise ValueError(f'Mismatch in sources between supply_capacity.csv and transportation_costs.csv: {set(sources) ^ set(cost_sources)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Mismatch in customers between customer_demand.csv and transportation_costs.csv: {set(customers) ^ set(cost_customers)}')
df_cost = df_cost.set_index('Unnamed: 0')
df_cost = df_cost.loc[sources, customers]
cost = {(s, c): float(df_cost.at[s, c]) for s in sources for c in customers}
m = gp.Model('Amazon_Transportation')
x = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Transportation Plan (quantity shipped from each source to each customer) ---')
    for s in sources:
        for c in customers:
            qty = x[s, c].X
            if qty > 1e-06:
                print(f'  From {s} to {c}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')