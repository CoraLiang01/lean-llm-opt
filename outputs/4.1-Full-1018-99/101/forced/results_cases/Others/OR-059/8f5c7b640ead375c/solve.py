import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
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
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
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
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_costs_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
for cust in customers:
    if cust not in trans_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing column for customer '{cust}'")
trans_cost_dict = {}
for idx, row in trans_costs_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        trans_cost_dict[sup, cust] = cost
if set(suppliers) != set(trans_costs_df['Unnamed: 0'].astype(str).str.strip()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(demand_dict.keys()):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv columns')
m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
trans_cost_expr = gp.quicksum((trans_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + trans_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()