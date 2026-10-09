import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv', dtype=str, keep_default_na=False)
warehouse_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv', dtype=str, keep_default_na=False)
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv', dtype=str, keep_default_na=False)
warehouse_ids = warehouse_df['Warehouse ID'].tolist()
customer_ids = demand_df['Customer ID'].tolist()
fixed_cost = {}
capacity = {}
for (idx, row) in warehouse_df.iterrows():
    wid = row['Warehouse ID']
    if wid not in warehouse_ids:
        raise ValueError(f'Warehouse ID {wid} in warehouse.csv not in warehouse_ids list.')
    try:
        fixed_cost[wid] = int(row['Fixed_Cost'])
        capacity[wid] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Fixed_Cost or Capacity for warehouse {wid}: {e}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cid = row['Customer ID']
    if cid not in customer_ids:
        raise ValueError(f'Customer ID {cid} in demand.csv not in customer_ids list.')
    try:
        demand[cid] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f'Error converting Demand for customer {cid}: {e}')
cost = {}
cost_customer_cols = [col for col in cost_df.columns if col != 'Warehouse ID']
for cid in customer_ids:
    if cid not in cost_customer_cols:
        raise ValueError(f'Customer ID {cid} not found as a column in cost.csv.')
for (idx, row) in cost_df.iterrows():
    wid = row['Warehouse ID']
    if wid not in warehouse_ids:
        raise ValueError(f'Warehouse ID {wid} in cost.csv not in warehouse_ids list.')
    for cid in customer_ids:
        try:
            cost[wid, cid] = int(row[cid])
        except Exception as e:
            raise ValueError(f'Error converting cost for warehouse {wid}, customer {cid}: {e}')
if set(warehouse_ids) != set(cost_df['Warehouse ID']):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv.')
if set(customer_ids) != set(cost_customer_cols):
    raise ValueError('Mismatch between customer IDs in demand.csv and cost.csv columns.')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars([(i, j) for i in warehouse_ids for j in customer_ids], vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((cost[i, j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i], name=f'capacity_{i}')
for i in warehouse_ids:
    for j in customer_ids:
        m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name=f'assign_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in warehouse_ids:
        print(f'{y_vars[i].VarName} {y_vars[i].X}')
    for i in warehouse_ids:
        for j in customer_ids:
            print(f'{x_vars[i, j].VarName} {x_vars[i, j].X}')
else:
    print(f'Solver status: {m.status}')