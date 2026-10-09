import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must have columns 'customer' and 'demand'")
customers = customer_demand_df['customer'].astype(str).str.strip()
customer_ids = customers.tolist()
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must have columns 'Unnamed: 0' and 'supply_capacity'")
warehouses = supply_capacity_df['Unnamed: 0'].astype(str).str.strip()
warehouse_ids = warehouses.tolist()
customer_demand_df = customer_demand_df.set_index(customer_demand_df['customer'].astype(str).str.strip())
demand_dict = customer_demand_df['demand'].astype(int).to_dict()
supply_capacity_df = supply_capacity_df.set_index(supply_capacity_df['Unnamed: 0'].astype(str).str.strip())
supply_capacity_dict = supply_capacity_df['supply_capacity'].astype(int).to_dict()
transportation_costs_df = transportation_costs_df.set_index(transportation_costs_df['Unnamed: 0'].astype(str).str.strip())
missing_warehouses = set(warehouse_ids) - set(transportation_costs_df.index)
if missing_warehouses:
    raise KeyError(f'Missing warehouses in transportation_costs.csv: {missing_warehouses}')
missing_customers = set(customer_ids) - set(transportation_costs_df.columns)
if missing_customers:
    raise KeyError(f'Missing customers in transportation_costs.csv: {missing_customers}')
cost_dict = {}
for s in warehouse_ids:
    for c in customer_ids:
        val = transportation_costs_df.loc[s, c]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Invalid cost value for warehouse '{s}', customer '{c}': '{val}'")
        cost_dict[s, c] = cost
m = gp.Model('TransportationProblem')
x_vars = m.addVars(warehouse_ids, customer_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in warehouse_ids for c in customer_ids)), gp.GRB.MINIMIZE)
for c in customer_ids:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in warehouse_ids)) == demand_dict[c], name=f'demand_{c}')
for s in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customer_ids)) <= supply_capacity_dict[s], name=f'supply_{s}')
m.optimize()