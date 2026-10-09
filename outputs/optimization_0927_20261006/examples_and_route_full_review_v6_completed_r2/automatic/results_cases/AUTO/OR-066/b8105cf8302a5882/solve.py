import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
if 'Unnamed: 0' not in transport_cost_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_suppliers = transport_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(suppliers) != set(transport_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(transport_suppliers)}')
transport_customer_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customer_cols):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customer_cols)}')
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = val
m = gp.Model('UFLP')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
total_transport = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('ACTIVATED' if y_vars[i].X > 0.5 else 'Not Activated')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x_{ij}) ---')
    for i in suppliers:
        for j in customers:
            if x_vars[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')