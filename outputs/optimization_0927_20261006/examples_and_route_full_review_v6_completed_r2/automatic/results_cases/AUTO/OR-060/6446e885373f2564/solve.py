import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customer_ids = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
supplier_ids = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
transport_supplier_ids = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(supplier_ids) != set(transport_supplier_ids):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
transport_customer_ids = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customer_ids) != set(transport_customer_ids):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    for cust in customer_ids:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost_dict[sup, cust] = val
m = gp.Model('UFLP')
y_vars = m.addVars(supplier_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(supplier_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in supplier_ids))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup, cust] * x_vars[sup, cust] for sup in supplier_ids for cust in customer_ids))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customer_ids:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in supplier_ids)) == demand_dict[cust], name=f'demand_{cust}')
for sup in supplier_ids:
    for cust in customer_ids:
        m.addConstr(x_vars[sup, cust] <= demand_dict[cust] * y_vars[sup], name=f'supply_link_{sup}_{cust}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Opening Decisions ---')
    for sup in supplier_ids:
        print(f"Supplier {sup}: {('OPEN' if y_vars[sup].X > 0.5 else 'closed')} (y={int(round(y_vars[sup].X))})")
    print('\n--- Supply Plan (x[i,j]) ---')
    for cust in customer_ids:
        print(f'Customer {cust} demand: {demand_dict[cust]}')
        for sup in supplier_ids:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplied by {sup}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')