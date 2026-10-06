import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', sep=',')
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', sep=',')
warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
customers = demand_df['customer'].astype(str).str.strip().tolist()
transport_warehouses = transport_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(warehouses) != set(transport_warehouses):
    raise ValueError(f'Mismatch in warehouse identifiers between fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(transport_warehouses)}')
transport_customers = [c for c in transport_df.columns if c != 'Unnamed: 0']
if set(customers) != set(transport_customers):
    raise ValueError(f'Mismatch in customer identifiers between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
transport_cost = {}
for (idx, row) in transport_df.iterrows():
    w = str(row['Unnamed: 0']).strip()
    for c in customers:
        transport_cost[w, c] = float(row[c])
for w in warehouses:
    if w not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {w}')
    for c in customers:
        if (w, c) not in transport_cost:
            raise ValueError(f'Missing transportation cost for warehouse {w}, customer {c}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
m = gp.Model('UFLP_Bandcamp')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x = m.addVars([(w, c) for w in warehouses for c in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[w] * y[w] for w in warehouses)) + gp.quicksum((transport_cost[w, c] * x[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
for w in warehouses:
    for c in customers:
        m.addConstr(x[w, c] <= demand[c] * y[w], name=f'link_{w}_{c}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for w in warehouses:
        print(f'{y[w].VarName} {y[w].X}')
    for w in warehouses:
        for c in customers:
            print(f'{x[w, c].VarName} {x[w, c].X}')
else:
    print(f'Solver status: {m.status}')