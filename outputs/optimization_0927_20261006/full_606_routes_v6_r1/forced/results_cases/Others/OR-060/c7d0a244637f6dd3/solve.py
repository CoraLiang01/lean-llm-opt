import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(float)
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = fixed_cost_df['supplier'].tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transport_df.columns:
    raise KeyError("transportation_costs.csv must have a supplier row label column 'Unnamed: 0'")
transport_df['supplier'] = transport_df['Unnamed: 0'].str.strip()
for c in customers:
    if c not in transport_df.columns:
        raise KeyError(f"Customer '{c}' not found as a column in transportation_costs.csv")
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    supplier = row['supplier']
    for c in customers:
        try:
            cost = float(row[c])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{c}': {row[c]}")
        transport_cost_dict[supplier, c] = cost
if set(suppliers) != set(transport_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) - set(demand_dict.keys()):
    raise ValueError('Some customers in demand.csv are missing demand values')
if set(customers) - set(transport_df.columns):
    raise ValueError('Some customers in demand.csv are missing in transportation_costs.csv columns')
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
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"Supplier {i}: {('OPEN' if y_vars[i].X > 0.5 else 'closed')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan ---')
    for j in customers:
        print(f'Customer {j} (demand={demand_dict[j]:.0f}):')
        for i in suppliers:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  From Supplier {i}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')