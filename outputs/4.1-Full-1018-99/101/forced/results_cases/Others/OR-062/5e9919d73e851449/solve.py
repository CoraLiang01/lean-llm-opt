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
if 'Customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must contain 'Customer' and 'demand' columns.")
customers = demand_df['Customer'].tolist()
try:
    demand_dict = {row['Customer']: int(row['demand']) for _, row in demand_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting demand values to int: {e}')
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must contain 'Unnamed: 0' and 'fixed_costs' columns.")
suppliers = [s.strip() for s in fixed_cost_df['Unnamed: 0'].tolist()]
try:
    fixed_cost_dict = {row['Unnamed: 0'].strip(): float(row['fixed_costs']) for _, row in fixed_cost_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting fixed_costs to float: {e}')
if 'Unnamed: 0' not in transport_df.columns:
    raise KeyError("transportation_costs.csv must contain 'Unnamed: 0' column for suppliers.")
transport_df['Supplier'] = transport_df['Unnamed: 0'].apply(lambda x: x.strip())
transport_suppliers = set(transport_df['Supplier'])
missing_suppliers = set(suppliers) - transport_suppliers
if missing_suppliers:
    raise ValueError(f'Suppliers {missing_suppliers} from fixed_cost.csv not found in transportation_costs.csv.')
transport_customer_cols = [col for col in transport_df.columns if col not in ['Unnamed: 0', 'Supplier']]
missing_customers = set(customers) - set(transport_customer_cols)
if missing_customers:
    raise ValueError(f'Customers {missing_customers} from demand.csv not found as columns in transportation_costs.csv.')
transport_cost_dict = {}
for _, row in transport_df.iterrows():
    supplier = row['Supplier']
    for customer in customers:
        try:
            cost_val = float(row[customer])
        except Exception as e:
            raise ValueError(f"Error converting transportation cost for supplier '{supplier}', customer '{customer}': {e}")
        transport_cost_dict[supplier, customer] = cost_val
supplier_set = suppliers
customer_set = customers
M = sum((demand_dict[c] for c in customer_set))
m = gp.Model('Iowa_Liquor_UFLP')
x_vars = m.addVars(supplier_set, customer_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(supplier_set, vtype=gp.GRB.BINARY, name='')
total_fixed_cost = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in supplier_set))
total_transport_cost = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in supplier_set for j in customer_set))
m.setObjective(total_fixed_cost + total_transport_cost, gp.GRB.MINIMIZE)
for j in customer_set:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in supplier_set)) == demand_dict[j], name='')
for i in supplier_set:
    for j in customer_set:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in supplier_set:
        print(f"  Supplier '{i}': {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Shipment Plan (supplier -> customer: quantity) ---')
    for i in supplier_set:
        for j in customer_set:
            if x_vars[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x_vars[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')