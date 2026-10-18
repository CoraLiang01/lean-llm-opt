import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must contain columns 'customer' and 'demand'")
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must contain columns 'Unnamed: 0' and 'fixed_costs'")
warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    wh = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost_dict[wh] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for warehouse {wh}: {row['fixed_costs']}")
if 'Unnamed: 0' not in transport_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for warehouse IDs")
for cust in customers:
    if cust not in transport_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing column for customer '{cust}'")
transport_cost_dict = {}
for idx, row in transport_costs_df.iterrows():
    wh = str(row['Unnamed: 0']).strip()
    for cust in customers:
        try:
            transport_cost_dict[wh, cust] = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}')
warehouses_from_transport = transport_costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(warehouses_from_transport) != set(warehouses):
    raise ValueError('Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv')
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transport_cost_dict[wh, cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand_dict[cust], name=f'demand_{cust}')
for wh in warehouses:
    for cust in customers:
        m.addConstr(x_vars[wh, cust] <= demand_dict[cust] * y_vars[wh], name=f'link_{wh}_{cust}')
m.optimize()