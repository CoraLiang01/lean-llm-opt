import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must contain columns 'customer' and 'demand'")
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
    raise KeyError("fixed_cost.csv must contain columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
if 'Unnamed: 0' not in transport_cost_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supplier IDs")
for cust in customers:
    if cust not in transport_cost_df.columns:
        raise KeyError(f"Customer '{cust}' not found as a column in transportation_costs.csv")
transport_cost_dict = {}
for idx, row in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        raise ValueError(f"Supplier '{sup}' in transportation_costs.csv not found in fixed_cost.csv")
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = cost
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup, cust] * x_vars[sup, cust] for sup in suppliers for cust in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand_dict[cust], name=f'demand_{cust}')
M = sum(demand_dict.values())
for sup in suppliers:
    for cust in customers:
        m.addConstr(x_vars[sup, cust] <= M * y_vars[sup], name=f'link_{sup}_{cust}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for sup in suppliers:
        status = 'ACTIVATED' if y_vars[sup].X > 0.5 else 'Not Activated'
        print(f'  Supplier {sup}: {status} (y={int(round(y_vars[sup].X))})')
    print('\n--- Supply Plan (x_{ij}) ---')
    for sup in suppliers:
        for cust in customers:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplier {sup} -> Customer {cust}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')