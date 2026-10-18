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
customers = demand_df['customer'].str.strip().tolist()
warehouses = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
transport_warehouse_ids = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(warehouses) != set(transport_warehouse_ids):
    raise ValueError('Mismatch between warehouse IDs in fixed_cost.csv and transportation_costs.csv')
transport_customer_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customer_cols):
    raise ValueError('Mismatch between customer IDs in demand.csv and transportation_costs.csv columns')
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[wh] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_cost value for warehouse {wh}: {row['fixed_costs']}") from e
transport_cost_dict = {}
for idx, row in transport_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            val = row[cust]
            transport_cost_dict[wh, cust] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}') from e
m = gp.Model('UFLP_Bandcamp')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in warehouses)) + gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouses)) == demand_dict[j])
for i in warehouses:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouse Activation ---')
    for i in warehouses:
        print(f"Warehouse {i}: {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('--- Customer Assignments ---')
    for j in customers:
        print(f'Customer {j}:')
        for i in warehouses:
            if x_vars[i, j].X > 1e-06:
                print(f'  From warehouse {i}: {x_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')