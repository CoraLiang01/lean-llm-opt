import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
demand_df['customer'] = demand_df['customer'].str.strip()
customers = demand_df['customer'].unique().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].str.strip()
suppliers = fixed_cost_df['supplier'].unique().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['supplier'].strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
transport_df['supplier'] = transport_df['Unnamed: 0'].str.strip()
transport_suppliers = transport_df['supplier'].unique().tolist()
transport_customers = [col for col in transport_df.columns if col not in ['Unnamed: 0', 'supplier']]
if set(suppliers) != set(transport_suppliers):
    raise ValueError(f'Mismatch in suppliers between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(transport_suppliers)}')
if set(customers) != set(transport_customers):
    raise ValueError(f'Mismatch in customers between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    sup = row['supplier'].strip()
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = val
M = sum((demand_dict[cust] for cust in customers))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x_keys = [(i, j) for i in suppliers for j in customers]
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()