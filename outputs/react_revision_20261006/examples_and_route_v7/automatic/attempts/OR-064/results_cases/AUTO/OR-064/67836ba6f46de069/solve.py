import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise ValueError("demand.csv must have columns 'customer' and 'demand'")
customers = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Non-integer demand for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Non-numeric fixed_costs for supplier {sup}: {row['fixed_costs']}")
    fixed_cost_dict[sup] = fc
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transport_df.columns:
    raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_suppliers = transport_df['Unnamed: 0'].str.strip().tolist()
transport_customers = [col for col in transport_df.columns if col != 'Unnamed: 0']
if set(suppliers) != set(transport_suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(transport_customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv columns')
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    for cust in customers:
        val = row[cust]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {val}')
        transport_cost_dict[sup, cust] = cost
supplier_set = suppliers
customer_set = customers

def solve_uflp(supplier_set, customer_set, fixed_cost_dict, demand_dict, transport_cost_dict):
    m = gp.Model('UFLP')
    y_vars = m.addVars(supplier_set, vtype=gp.GRB.BINARY, name='')
    x_vars = m.addVars([(i, j) for i in supplier_set for j in customer_set], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_fixed = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in supplier_set))
    total_transport = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in supplier_set for j in customer_set))
    m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
    for j in customer_set:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in supplier_set)) == demand_dict[j], name=f'demand_{j}')
    for i in supplier_set:
        for j in customer_set:
            m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'activate_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_uflp(supplier_set, customer_set, fixed_cost_dict, demand_dict, transport_cost_dict)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')