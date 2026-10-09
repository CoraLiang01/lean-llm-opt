import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
warehouse_ids = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
transport_warehouse_ids = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(warehouse_ids) != set(transport_warehouse_ids):
    raise ValueError('Mismatch between warehouse IDs in fixed_cost.csv and transportation_costs.csv')
customer_ids = demand_df['customer'].str.strip().tolist()
transport_customer_ids = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customer_ids) != set(transport_customer_ids):
    raise ValueError('Mismatch between customer IDs in demand.csv and transportation_costs.csv columns')
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    wid = row['Unnamed: 0'].strip()
    try:
        fixed_costs[wid] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for warehouse {wid}: {row['fixed_costs']}") from e
demands = {}
for (idx, row) in demand_df.iterrows():
    cid = row['customer'].strip()
    try:
        demands[cid] = float(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cid}: {row['demand']}") from e
transport_costs = {}
for (idx, row) in transport_cost_df.iterrows():
    wid = row['Unnamed: 0'].strip()
    for cid in customer_ids:
        try:
            transport_costs[wid, cid] = float(row[cid])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wid}, customer {cid}: {row[cid]}') from e
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_costs[wid] * y_vars[wid] for wid in warehouse_ids)) + gp.quicksum((transport_costs[wid, cid] * x_vars[wid, cid] for wid in warehouse_ids for cid in customer_ids)), gp.GRB.MINIMIZE)
for cid in customer_ids:
    m.addConstr(gp.quicksum((x_vars[wid, cid] for wid in warehouse_ids)) == demands[cid], name=f'demand_{cid}')
for wid in warehouse_ids:
    for cid in customer_ids:
        m.addConstr(x_vars[wid, cid] <= demands[cid] * y_vars[wid], name=f'link_{wid}_{cid}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouse Activation ---')
    for wid in warehouse_ids:
        print(f"  Warehouse {wid}: {('OPEN' if y_vars[wid].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[wid].X))})")
    print('--- Shipment Plan ---')
    for cid in customer_ids:
        print(f'  Customer {cid}:')
        for wid in warehouse_ids:
            shipped = x_vars[wid, cid].X
            if shipped > 1e-06:
                print(f'    From warehouse {wid}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')