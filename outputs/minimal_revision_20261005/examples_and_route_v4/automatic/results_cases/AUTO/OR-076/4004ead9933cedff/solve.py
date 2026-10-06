import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
warehouse_df = pd.read_csv(warehouse_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
cost_df['Warehouse ID'] = cost_df['Warehouse ID'].astype(str).str.strip()
warehouse_df['Warehouse ID'] = warehouse_df['Warehouse ID'].astype(str).str.strip()
demand_df['Customer ID'] = demand_df['Customer ID'].astype(str).str.strip()
warehouses = list(warehouse_df['Warehouse ID'].unique())
customers = list(demand_df['Customer ID'].unique())
cost_warehouses = set(cost_df['Warehouse ID'])
if set(warehouses) != cost_warehouses:
    raise ValueError(f'Mismatch in warehouse IDs between warehouse.csv and cost.csv: {set(warehouses)} vs {cost_warehouses}')
cost_customers = [col for col in cost_df.columns if col.startswith('C')]
if set(customers) != set(cost_customers):
    raise ValueError(f'Mismatch in customer IDs between demand.csv and cost.csv: {set(customers)} vs {set(cost_customers)}')
fixed_cost = warehouse_df.set_index('Warehouse ID')['Fixed_Cost'].to_dict()
capacity = warehouse_df.set_index('Warehouse ID')['Capacity'].to_dict()
demand = demand_df.set_index('Customer ID')['Demand'].to_dict()
cost = {}
for (_, row) in cost_df.iterrows():
    w = row['Warehouse ID']
    cost[w] = {}
    for c in customers:
        val = row[c]
        if not np.isfinite(val):
            raise ValueError(f'Missing or invalid cost for warehouse {w}, customer {c}')
        cost[w][c] = float(val)
for w in warehouses:
    if w not in fixed_cost or w not in capacity or w not in cost:
        raise ValueError(f'Missing warehouse data for {w}')
    for c in customers:
        if c not in cost[w]:
            raise ValueError(f'Missing cost for warehouse {w}, customer {c}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
m = gp.Model('UFLP')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x = m.addVars([(w, c) for w in warehouses for c in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[w] * y[w] for w in warehouses)) + gp.quicksum((cost[w][c] * x[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, c] for c in customers)) <= capacity[w] * y[w], name=f'capacity_{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for w in warehouses:
        print(f'y[{w}] {y[w].VarName} {y[w].X}')
    for w in warehouses:
        for c in customers:
            print(f'x[{w},{c}] {x[w, c].VarName} {x[w, c].X}')
else:
    print(f'Solver status: {m.status}')