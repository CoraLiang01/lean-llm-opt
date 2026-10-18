import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
demand_df['demand'] = demand_df['demand'].astype(float)
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transport_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
for cust in customers:
    if cust not in transport_df.columns:
        raise KeyError(f"Customer '{cust}' not found as a column in transportation_costs.csv")
for cust in customers:
    transport_df[cust] = transport_df[cust].astype(float)
supplier_ids_transport = transport_df['Unnamed: 0'].tolist()
if set(suppliers) != set(supplier_ids_transport):
    raise ValueError('Supplier IDs in fixed_cost.csv and transportation_costs.csv do not match exactly')
transport_cost_dict = {}
for idx, row in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    for cust in customers:
        transport_cost_dict[supplier, cust] = float(row[cust])
if len(suppliers) != len(set(suppliers)):
    raise ValueError('Duplicate supplier IDs found in fixed_cost.csv')
if len(customers) != len(set(customers)):
    raise ValueError('Duplicate customer IDs found in demand.csv')
if set(suppliers) != set(supplier_ids_transport):
    raise ValueError('Mismatch in supplier IDs between fixed_cost.csv and transportation_costs.csv')
for i, j in [(i, j) for i in suppliers for j in customers]:
    if (i, j) not in transport_cost_dict:
        raise KeyError(f'Missing transportation cost for supplier {i}, customer {j}')
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'activate_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Opened Suppliers ---')
    for i in suppliers:
        if y_vars[i].X > 0.5:
            print(f'  Supplier {i}: OPEN (fixed cost {fixed_cost_dict[i]:.2f})')
    print('\n--- Supply Plan (nonzero shipments) ---')
    for i in suppliers:
        for j in customers:
            if x_vars[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x_vars[i, j].X:.2f} units (cost per unit {transport_cost_dict[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')