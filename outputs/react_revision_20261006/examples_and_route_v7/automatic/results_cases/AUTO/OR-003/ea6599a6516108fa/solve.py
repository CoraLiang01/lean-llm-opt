import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
customer_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_df.columns or 'demand' not in customer_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'")
customers = customer_df['customer'].astype(str).tolist()
customer_demand = {}
for (idx, row) in customer_df.iterrows():
    cust = str(row['customer'])
    try:
        demand = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    customer_demand[cust] = demand
supply_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in supply_df.columns or 'supply_capacity' not in supply_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
suppliers = supply_df['Unnamed: 0'].astype(str).tolist()
supply_capacity = {}
for (idx, row) in supply_df.iterrows():
    sup = str(row['Unnamed: 0'])
    try:
        cap = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
    supply_capacity[sup] = cap
costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supplier IDs")
cost_columns = [col for col in costs_df.columns if col != 'Unnamed: 0']
missing_customers = set(customers) - set(cost_columns)
if missing_customers:
    raise KeyError(f'transportation_costs.csv missing cost columns for customers: {missing_customers}')
cost_supplier_ids = costs_df['Unnamed: 0'].astype(str).tolist()
missing_suppliers = set(suppliers) - set(cost_supplier_ids)
if missing_suppliers:
    raise KeyError(f'transportation_costs.csv missing rows for suppliers: {missing_suppliers}')
transportation_cost = {}
for (idx, row) in costs_df.iterrows():
    sup = str(row['Unnamed: 0'])
    if sup not in suppliers:
        continue
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transportation_cost[sup, cust] = cost
for s in suppliers:
    for c in customers:
        if (s, c) not in transportation_cost:
            raise KeyError(f'Missing transportation cost for supplier {s}, customer {c}')

def solve_transportation_problem(suppliers, customers, supply_capacity, customer_demand, transportation_cost):
    m = gp.Model('Transportation')
    keys = [(s, c) for s in suppliers for c in customers]
    x_vars = m.addVars(keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((transportation_cost[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
    for s in suppliers:
        m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
    for c in customers:
        m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == customer_demand[c], name=f'demand_{c}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_transportation_problem(suppliers, customers, supply_capacity, customer_demand, transportation_cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')