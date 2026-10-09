import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
if 'Unnamed: 0' not in transport_cost_df.columns:
    raise KeyError("transportation_costs.csv must have 'Unnamed: 0' as supplier row labels")
transport_suppliers = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(suppliers) != set(transport_suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
transport_customer_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customer_cols):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = val
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
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
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Opened Suppliers ---')
    for sup in suppliers:
        if y_vars[sup].X > 0.5:
            print(f'  {sup}: OPEN (fixed cost {fixed_cost_dict[sup]:.2f})')
    print('\n--- Supply Plan (x_{i,j} > 0) ---')
    for sup in suppliers:
        for cust in customers:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplier {sup} -> Customer {cust}: {val:.2f} units (cost per unit: {transport_cost_dict[sup, cust]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')