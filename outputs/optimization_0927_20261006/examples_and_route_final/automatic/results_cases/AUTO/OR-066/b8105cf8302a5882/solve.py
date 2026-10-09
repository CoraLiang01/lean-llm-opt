import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['customer'].astype(str).tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer'])
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
suppliers_fc = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
suppliers_tc = transport_cost_df['Unnamed: 0'].astype(str).tolist()
if set(suppliers_fc) != set(suppliers_tc):
    raise ValueError(f'Supplier sets in fixed_cost.csv and transportation_costs.csv do not match: {suppliers_fc} vs {suppliers_tc}')
suppliers = suppliers_fc
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0'])
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0'])
    for cust in customers:
        try:
            tc = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = tc
F = suppliers
C = customers
m = gp.Model('UFLP')
x_vars = m.addVars(F, C, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(F, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in F))
transport_cost_expr = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in F for j in C))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in C:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in F)) == demand_dict[j], name=f'demand_{j}')
M = sum((demand_dict[j] for j in C))
for i in F:
    for j in C:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in F:
        print(f"  Supplier {i}: {('ACTIVATED' if y_vars[i].X > 0.5 else 'Not activated')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x[i,j]) ---')
    for i in F:
        for j in C:
            if x_vars[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')