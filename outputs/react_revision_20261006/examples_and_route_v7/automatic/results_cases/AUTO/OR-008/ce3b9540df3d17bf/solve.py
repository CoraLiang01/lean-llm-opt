import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return re.sub('\\s+', '', str(x)).casefold()

def solve_freshmart_transportation():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv', dtype=str, keep_default_na=False)
    supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv', dtype=str, keep_default_na=False)
    cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', dtype=str, keep_default_na=False)
    customers = demand_df['Customers'].map(normalize_id).tolist()
    customer_id_map = dict(zip(demand_df['Customers'].map(normalize_id), demand_df['Customers']))
    suppliers = supply_df['Suppliers'].map(normalize_id).tolist()
    supplier_id_map = dict(zip(supply_df['Suppliers'].map(normalize_id), supply_df['Suppliers']))
    cost_supplier_ids = cost_df['Unnamed: 0'].map(normalize_id).tolist()
    cost_customer_ids = [normalize_id(c) for c in cost_df.columns if c != 'Unnamed: 0']
    if set(suppliers) != set(cost_supplier_ids):
        raise ValueError(f'Mismatch in suppliers between supply_capacity.csv and transportation_costs.csv: {set(suppliers)} vs {set(cost_supplier_ids)}')
    if set(customers) != set(cost_customer_ids):
        raise ValueError(f'Mismatch in customers between customer_demand.csv and transportation_costs.csv: {set(customers)} vs {set(cost_customer_ids)}')
    demand_dict = {}
    for (idx, row) in demand_df.iterrows():
        cust_norm = normalize_id(row['Customers'])
        try:
            demand_dict[cust_norm] = int(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {row['Customers']}: {row['demand']}")
    supply_dict = {}
    for (idx, row) in supply_df.iterrows():
        supp_norm = normalize_id(row['Suppliers'])
        try:
            supply_dict[supp_norm] = int(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {row['Suppliers']}: {row['supply_capacity']}")
    cost_dict = {}
    for (i, row) in cost_df.iterrows():
        supp_norm = normalize_id(row['Unnamed: 0'])
        for cust_col in cost_df.columns:
            if cust_col == 'Unnamed: 0':
                continue
            cust_norm = normalize_id(cust_col)
            try:
                cost_dict[supp_norm, cust_norm] = float(row[cust_col])
            except Exception:
                raise ValueError(f"Invalid transportation cost for supplier {row['Unnamed: 0']} to customer {cust_col}: {row[cust_col]}")
    for s in suppliers:
        for c in customers:
            if (s, c) not in cost_dict:
                raise ValueError(f'Missing transportation cost for supplier {supplier_id_map[s]} to customer {customer_id_map[c]}')
    m = gp.Model('FreshMart_Transportation')
    shipment_keys = [(s, c) for s in suppliers for c in customers]
    shipment_vars = m.addVars(shipment_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost_dict[s, c] * shipment_vars[s, c] for (s, c) in shipment_keys)), gp.GRB.MINIMIZE)
    for s in suppliers:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for c in customers)) <= supply_dict[s], name='supply_' + s)
    for c in customers:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for s in suppliers)) == demand_dict[c], name='demand_' + c)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in suppliers:
            for c in customers:
                var = shipment_vars[s, c]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_freshmart_transportation()