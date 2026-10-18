import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
if 'Unnamed: 0' not in transport_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_df = transport_df.set_index('Unnamed: 0')
transport_df.index = transport_df.index.astype(str).str.strip()
missing_suppliers = set(suppliers) - set(transport_df.index)
if missing_suppliers:
    raise KeyError(f'transportation_costs.csv missing suppliers: {missing_suppliers}')
missing_customers = set(customers) - set(transport_df.columns)
if missing_customers:
    raise KeyError(f'transportation_costs.csv missing customers: {missing_customers}')
transport_cost_dict = {}
for sup in suppliers:
    for cust in customers:
        val = transport_df.loc[sup, cust]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {val}')
        transport_cost_dict[sup, cust] = cost
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[sup] * y_vars[sup] for sup in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[sup, cust] * x_vars[sup, cust] for sup in suppliers for cust in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand_dict[cust])
for sup in suppliers:
    for cust in customers:
        m.addConstr(x_vars[sup, cust] <= demand_dict[cust] * y_vars[sup])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for sup in suppliers:
        print(f"  Supplier {sup}: {('OPEN' if y_vars[sup].X > 0.5 else 'closed')} (y={int(round(y_vars[sup].X))})")
    print('\n--- Supply Plan (x_ij > 0) ---')
    for sup in suppliers:
        for cust in customers:
            val = x_vars[sup, cust].X
            if val > 1e-06:
                print(f'  Supplier {sup} -> Customer {cust}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')