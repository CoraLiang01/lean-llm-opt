import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise ValueError("demand.csv must have columns 'customer' and 'demand'.")
demand_df['customer'] = demand_df['customer'].str.strip()
dealerships = demand_df['customer'].tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'.")
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].str.strip()
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_cost_df.columns:
    raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'.")
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].str.strip()
for cust in dealerships:
    if cust not in trans_cost_df.columns:
        raise ValueError(f'Dealership {cust} not found as a column in transportation_costs.csv.')
trans_cost_dict = {}
for idx, row in trans_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in transportation_costs.csv not found in fixed_cost.csv.')
    for cust in dealerships:
        try:
            trans_cost_dict[sup, cust] = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, dealership {cust}: {row[cust]}')
m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, dealerships, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
trans_cost_expr = gp.quicksum((trans_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in dealerships))
m.setObjective(fixed_cost_expr + trans_cost_expr, gp.GRB.MINIMIZE)
for j in dealerships:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in dealerships:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"Supplier {i}: {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Dealership Supply Plan ---')
    for j in dealerships:
        print(f'Dealership {j}:')
        for i in suppliers:
            qty = x_vars[i, j].X
            if qty > 0.001:
                print(f'  Supplied by {i}: {qty:.0f} vehicles')
else:
    print(f'No optimal solution found. Status: {m.status}')