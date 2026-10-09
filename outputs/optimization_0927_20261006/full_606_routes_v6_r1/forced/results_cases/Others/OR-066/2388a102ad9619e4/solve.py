import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
suppliers_fixed = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
suppliers_trans = transport_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(suppliers_fixed) != set(suppliers_trans):
    raise ValueError(f'Supplier sets in fixed_cost.csv and transportation_costs.csv do not match: {suppliers_fixed} vs {suppliers_trans}')
suppliers = suppliers_fixed
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}") from e
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    transport_cost_dict[sup] = {}
    for cust in customers:
        if cust not in row:
            raise ValueError(f'Customer {cust} not found in transportation_costs.csv columns')
        try:
            transport_cost_dict[sup][cust] = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}') from e
m = gp.Model('UFLP')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand_dict[cust], name=f'demand_{cust}')
for sup in suppliers:
    for cust in customers:
        m.addConstr(x_vars[sup, cust] <= demand_dict[cust] * y_vars[sup], name=f'link_{sup}_{cust}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for sup in suppliers:
        print(f"  Supplier {sup}: {('ACTIVATED' if y_vars[sup].X > 0.5 else 'Not activated')} (y={int(round(y_vars[sup].X))})")
    print('\n--- Supply Plan ---')
    for sup in suppliers:
        for cust in customers:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplier {sup} -> Customer {cust}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')