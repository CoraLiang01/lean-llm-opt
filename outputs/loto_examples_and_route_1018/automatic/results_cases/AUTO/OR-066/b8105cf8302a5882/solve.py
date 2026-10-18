import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
if 'Unnamed: 0' not in transport_costs_df.columns:
    raise KeyError("transportation_costs.csv must have 'Unnamed: 0' as supplier key column")
for cust in customers:
    if cust not in transport_costs_df.columns:
        raise KeyError(f'Customer {cust} not found as column in transportation_costs.csv')
transport_cost_dict = {}
for idx, row in transport_costs_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in transportation_costs.csv not found in fixed_cost.csv')
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = cost
if set(suppliers) != set(fixed_cost_dict.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and fixed_cost_dict')
if set(customers) != set(demand_dict.keys()):
    raise ValueError('Mismatch between customers in demand.csv and demand_dict')
for sup in suppliers:
    for cust in customers:
        if (sup, cust) not in transport_cost_dict:
            raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
m = gp.Model('UFLP')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed_cost = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
total_transport_cost = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed_cost + total_transport_cost, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
M = sum(demand_dict.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('ACTIVATED' if y_vars[i].X > 0.5 else 'NOT activated')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x_{ij}) ---')
    for i in suppliers:
        for j in customers:
            if x_vars[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')