import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = [c.strip() for c in demand_df['customer']]
transport_customers = [c.strip() for c in transport_cost_df.columns if c.strip() != 'Unnamed: 0']
if set(customers) != set(transport_customers):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
suppliers = [s.strip() for s in fixed_cost_df['Unnamed: 0']]
transport_suppliers = [str(s).strip() for s in transport_cost_df['Unnamed: 0']]
if set(suppliers) != set(transport_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(transport_suppliers)}')
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}") from e
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}') from e
        transport_cost_dict[sup, cust] = val
M = sum((demand_dict[cust] for cust in customers))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()