import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Customers' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain 'Customers' and 'demand' columns.")
customers = customer_demand_df['Customers'].astype(str).tolist()
if 'Suppliers' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain 'Suppliers' and 'supply_capacity' columns.")
suppliers = supply_capacity_df['Suppliers'].astype(str).tolist()
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain 'Unnamed: 0' column for supplier IDs.")
transportation_costs_df['Supplier_ID'] = transportation_costs_df['Unnamed: 0'].apply(lambda x: x.strip())
cost_matrix_suppliers = transportation_costs_df['Supplier_ID'].tolist()
cost_matrix_customers = [col for col in transportation_costs_df.columns if col not in ['Unnamed: 0', 'Supplier_ID']]
missing_suppliers = set(suppliers) - set(cost_matrix_suppliers)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
missing_customers = set(customers) - set(cost_matrix_customers)
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
demand_dict = {}
for (idx, row) in customer_demand_df.iterrows():
    cust = str(row['Customers'])
    try:
        demand = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand
supply_capacity_dict = {}
for (idx, row) in supply_capacity_df.iterrows():
    supp = str(row['Suppliers'])
    try:
        cap = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for supplier {supp}: {row['supply_capacity']}")
    supply_capacity_dict[supp] = cap
cost_dict = {}
for (_, row) in transportation_costs_df.iterrows():
    supp = row['Supplier_ID']
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {supp}, customer {cust}: {row[cust]}')
        cost_dict[supp, cust] = cost
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity_dict[s])
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c])
m.optimize()