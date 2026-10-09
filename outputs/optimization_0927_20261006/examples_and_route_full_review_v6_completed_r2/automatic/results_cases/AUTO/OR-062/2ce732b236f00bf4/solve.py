import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['Customer'].tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['Customer']
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer '{cust}': {row['demand']}")
    demand_dict[cust] = demand_val
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    supplier = row['Unnamed: 0']
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for supplier '{supplier}': {row['fixed_costs']}")
    fixed_cost_dict[supplier] = fc
transport_df['Unnamed: 0'] = transport_df['Unnamed: 0'].apply(lambda x: x.strip())
transport_customer_cols = [col for col in transport_df.columns if col != 'Unnamed: 0']
if len(customers) != len(transport_customer_cols):
    raise ValueError('Number of customers in demand.csv does not match number of customer columns in transportation_costs.csv.')
customer_to_col = dict(zip(customers, transport_customer_cols))
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    for cust in customers:
        col = customer_to_col[cust]
        try:
            cost = float(row[col])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{cust}' (column '{col}'): {row[col]}")
        transport_cost_dict[supplier, cust] = cost
m = gp.Model('UFLP_Iowa_Liquor')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed_cost = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
total_transport_cost = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed_cost + total_transport_cost, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
M = sum(demand_dict.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in suppliers:
        status = 'OPEN' if y_vars[i].X > 0.5 else 'CLOSED'
        print(f"  Supplier '{i}': {status} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x_{ij}) ---')
    for i in suppliers:
        for j in customers:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f"  Supplier '{i}' -> Customer '{j}': {qty:.2f}")
else:
    print(f'No optimal solution found. Status: {m.status}')