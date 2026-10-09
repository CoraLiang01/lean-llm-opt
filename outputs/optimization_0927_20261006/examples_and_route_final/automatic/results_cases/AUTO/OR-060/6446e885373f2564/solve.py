import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
suppliers = list(fixed_cost_df['Unnamed: 0'].str.strip())
customers = list(demand_df['customer'].str.strip())
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for supplier {sup}: {row['fixed_costs']}") from e
transport_cost_dict = {}
transport_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
missing_customers = set(customers) - set(transport_cols)
if missing_customers:
    raise KeyError(f'Customers {missing_customers} not found as columns in transportation_costs.csv')
for (idx, row) in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        continue
    for cust in customers:
        try:
            val = row[cust]
            transport_cost_dict[sup, cust] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}') from e
for sup in suppliers:
    if sup not in fixed_cost_dict:
        raise KeyError(f'Supplier {sup} missing in fixed_cost.csv')
    for cust in customers:
        if (sup, cust) not in transport_cost_dict:
            raise KeyError(f'Missing transportation cost for supplier {sup}, customer {cust}')
for cust in customers:
    if cust not in demand_dict:
        raise KeyError(f'Customer {cust} missing in demand.csv')
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup, cust] * x_vars[sup, cust] for sup in suppliers for cust in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand_dict[cust], name=f'demand_{cust}')
for sup in suppliers:
    for cust in customers:
        m.addConstr(x_vars[sup, cust] <= demand_dict[cust] * y_vars[sup], name=f'link_{sup}_{cust}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Opened Suppliers ---')
    for sup in suppliers:
        if y_vars[sup].X > 0.5:
            print(f'  {sup}: OPEN (fixed cost = {fixed_cost_dict[sup]:.2f})')
    print('\n--- Supply Plan (x_ij > 0) ---')
    for sup in suppliers:
        for cust in customers:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplier {sup} -> Customer {cust}: {val:.2f} units (transport cost per unit = {transport_cost_dict[sup, cust]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')