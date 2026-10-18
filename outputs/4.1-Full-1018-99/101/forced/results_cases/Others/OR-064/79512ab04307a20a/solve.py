import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
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
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_df['supplier'] = transport_df['Unnamed: 0'].str.strip()
for cust in customers:
    if cust not in transport_df.columns:
        raise KeyError(f"Customer '{cust}' not found as a column in transportation_costs.csv")
transport_cost_dict = {}
for idx, row in transport_df.iterrows():
    supplier = row['supplier']
    if supplier not in suppliers:
        continue
    for cust in customers:
        val = row[cust]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{cust}': '{val}'")
        transport_cost_dict[supplier, cust] = cost
if set(suppliers) != set(transport_df['supplier']):
    missing = set(suppliers) - set(transport_df['supplier'])
    if missing:
        raise KeyError(f'Suppliers {missing} in fixed_cost.csv not found in transportation_costs.csv')
if set(customers) - set(demand_df['customer']):
    raise KeyError('Mismatch in customers between demand.csv and transportation_costs.csv')
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[s] * y_vars[s] for s in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[s, c] * x_vars[s, c] for s in suppliers for c in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
for s in suppliers:
    for c in customers:
        m.addConstr(x_vars[s, c] <= demand_dict[c] * y_vars[s], name=f'link_{s}_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for s in suppliers:
        print(f"Supplier {s}: {('OPEN' if y_vars[s].X > 0.5 else 'closed')} (y={int(round(y_vars[s].X))})")
    print('\n--- Supply Plan (nonzero flows) ---')
    for s in suppliers:
        for c in customers:
            val = x_vars[s, c].X
            if val > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')