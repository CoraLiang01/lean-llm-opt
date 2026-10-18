import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].str.strip().tolist()
if len(set(customers)) != len(customers):
    raise ValueError('Duplicate customer IDs found in demand.csv')
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
if len(set(suppliers)) != len(suppliers):
    raise ValueError('Duplicate supplier IDs found in fixed_cost.csv')
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
if 'Unnamed: 0' not in transport_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_cost_dict = {}
for idx, row in transport_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in transportation_costs.csv not found in fixed_cost.csv')
    for cust in customers:
        if cust not in transport_df.columns:
            raise KeyError(f'Customer {cust} not found as column in transportation_costs.csv')
        try:
            tc = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = tc
I = suppliers
J = customers
M = sum((demand_dict[cust] for cust in J))
m = gp.Model('UFLP')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(I, J, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in I))
transport_cost_expr = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in I for j in J))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == demand_dict[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in I:
        print(f"Supplier {i}: {('OPEN' if y_vars[i].X > 0.5 else 'closed')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x[i,j]) ---')
    for j in J:
        print(f'Customer {j}:')
        for i in I:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  Supplied by {i}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')