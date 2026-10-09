import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', dtype=str, keep_default_na=False)
warehouse_ids = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
customer_ids = demand_df['customer'].str.strip().tolist()
transport_warehouses = transport_df['Unnamed: 0'].str.strip().tolist()
if set(warehouse_ids) != set(transport_warehouses):
    raise ValueError(f'Mismatch in warehouse IDs between fixed_cost.csv and transportation_costs.csv: {warehouse_ids} vs {transport_warehouses}')
transport_customers = [col for col in transport_df.columns if col != 'Unnamed: 0']
if set(customer_ids) != set(transport_customers):
    raise ValueError(f'Mismatch in customer IDs between demand.csv and transportation_costs.csv: {customer_ids} vs {transport_customers}')
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    wid = row['Unnamed: 0'].strip()
    try:
        fixed_costs[wid] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for warehouse {wid}: {row['fixed_costs']}")
demands = {}
for (idx, row) in demand_df.iterrows():
    cid = row['customer'].strip()
    try:
        demands[cid] = float(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cid}: {row['demand']}")
transport_costs = {}
for (idx, row) in transport_df.iterrows():
    wid = row['Unnamed: 0'].strip()
    for cid in customer_ids:
        try:
            transport_costs[wid, cid] = float(row[cid])
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {wid}, customer {cid}: {row[cid]}')
for wid in warehouse_ids:
    if wid not in fixed_costs:
        raise ValueError(f'Missing fixed cost for warehouse {wid}')
    for cid in customer_ids:
        if (wid, cid) not in transport_costs:
            raise ValueError(f'Missing transportation cost for warehouse {wid}, customer {cid}')
for cid in customer_ids:
    if cid not in demands:
        raise ValueError(f'Missing demand for customer {cid}')
warehouse_set = warehouse_ids
customer_set = customer_ids
x_keys = [(i, j) for i in warehouse_set for j in customer_set]
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouse_set, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(x_keys, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in warehouse_set)) + gp.quicksum((transport_costs[i, j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
for j in customer_set:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_set)) >= demands[j], name=f'demand_{j}')
for i in warehouse_set:
    for j in customer_set:
        m.addConstr(x_vars[i, j] <= demands[j] * y_vars[i], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in warehouse_set:
        print(f'{y_vars[i].VarName} {y_vars[i].X}')
    for (i, j) in x_keys:
        print(f'{x_vars[i, j].VarName} {x_vars[i, j].X}')
else:
    print(f'Solver status: {m.Status}')