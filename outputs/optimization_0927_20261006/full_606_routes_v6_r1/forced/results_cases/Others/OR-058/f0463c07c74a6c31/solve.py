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
customers = demand_df['customer'].str.strip().tolist()
transport_customer_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customer_cols):
    raise ValueError(f'Mismatch between customers in demand.csv and columns in transportation_costs.csv: {set(customers)} vs {set(transport_customer_cols)}')
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
transport_supplier_rows = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(suppliers) != set(transport_supplier_rows):
    raise ValueError(f'Mismatch between suppliers in fixed_cost.csv and rows in transportation_costs.csv: {set(suppliers)} vs {set(transport_supplier_rows)}')
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            tc = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = tc
I = suppliers
J = customers
M = sum((demand_dict[j] for j in J))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x_vars = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in I)) + gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == demand_dict[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()