import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['Customer'].tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['Customer']
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer '{cust}': {row['demand']}")
    demand_dict[cust] = demand_val
suppliers = [s.strip() for s in fixed_cost_df['Unnamed: 0'].tolist()]
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    supplier = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier '{supplier}': {row['fixed_costs']}")
    fixed_cost_dict[supplier] = fc
transport_suppliers = [s.strip() for s in transport_df['Unnamed: 0'].tolist()]
transport_store_cols = [col for col in transport_df.columns if col != 'Unnamed: 0']

def normalize_name(s):
    return s.strip().casefold()
customer_norm_map = {normalize_name(cust): cust for cust in customers}
store_col_to_customer = {}
for col in transport_store_cols:
    norm_col = normalize_name(col)
    if norm_col not in customer_norm_map:
        raise ValueError(f"Store column '{col}' in transportation_costs.csv does not match any customer in demand.csv")
    store_col_to_customer[col] = customer_norm_map[norm_col]
transport_cost_dict = {}
for idx, row in transport_df.iterrows():
    supplier = row['Unnamed: 0'].strip()
    if supplier not in suppliers:
        raise ValueError(f"Supplier '{supplier}' in transportation_costs.csv not found in fixed_cost.csv")
    for col in transport_store_cols:
        customer = store_col_to_customer[col]
        try:
            cost = float(row[col])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{customer}': {row[col]}")
        transport_cost_dict[supplier, customer] = cost
if set(transport_suppliers) != set(suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(store_col_to_customer.values()) != set(customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv columns')
I = suppliers
J = customers
M = sum((demand_dict[j] for j in J))
m = gp.Model('UFLP_Iowa_Liquor')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
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
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in I:
        print(f"  Supplier '{i}': {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Shipment Plan (units from supplier to customer) ---')
    for i in I:
        for j in J:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f"  {qty:.2f} units from '{i}' to '{j}'")
else:
    print(f'No optimal solution found. Status: {m.status}')