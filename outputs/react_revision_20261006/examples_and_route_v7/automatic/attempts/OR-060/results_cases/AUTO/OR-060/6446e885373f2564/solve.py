import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise ValueError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].astype(str).tolist()
demand_dict = dict(zip(customers, demand_df['demand'].astype(float)))
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
fixed_cost_dict = dict(zip(suppliers, fixed_cost_df['fixed_costs'].astype(float)))
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transport_df.columns:
    raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_customer_cols = [col for col in transport_df.columns if col != 'Unnamed: 0']
missing_customers = set(customers) - set(transport_customer_cols)
if missing_customers:
    raise ValueError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
missing_suppliers = set(suppliers) - set(transport_df['Unnamed: 0'].astype(str))
if missing_suppliers:
    raise ValueError(f'transportation_costs.csv missing rows for suppliers: {missing_suppliers}')
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    supplier = str(row['Unnamed: 0'])
    for customer in customers:
        val = row[customer]
        try:
            transport_cost_dict[supplier, customer] = float(val)
        except Exception:
            raise ValueError(f'Non-numeric or missing transportation cost for supplier {supplier}, customer {customer}')
for i in suppliers:
    for j in customers:
        if (i, j) not in transport_cost_dict:
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')

def solve_uflp(suppliers, customers, fixed_cost_dict, demand_dict, transport_cost_dict):
    m = gp.Model('UFLP')
    x_vars = m.addVars([(i, j) for i in suppliers for j in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
    for i in suppliers:
        for j in customers:
            m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_uflp(suppliers, customers, fixed_cost_dict, demand_dict, transport_cost_dict)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')