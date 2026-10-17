import gurobipy as gp
from gurobipy import GRB
import pandas as pd
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv', dtype=str, keep_default_na=False)
if demand_df.shape[1] != 2:
    raise ValueError('demand.csv must have exactly two columns: customer and demand')
demand = {}
customers = []
for idx, row in demand_df.iterrows():
    cust = row[0]
    try:
        val = float(row[1])
    except Exception:
        raise ValueError(f'Non-numeric demand for customer {cust}')
    demand[cust] = val
    customers.append(cust)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv', dtype=str, keep_default_na=False)
if fixed_cost_df.shape[1] != 2:
    raise ValueError('fixed_cost.csv must have exactly two columns: facility and fixed_cost')
fixed_cost = {}
warehouses = []
for idx, row in fixed_cost_df.iterrows():
    wh = row[0]
    try:
        val = float(row[1])
    except Exception:
        raise ValueError(f'Non-numeric fixed cost for warehouse {wh}')
    fixed_cost[wh] = val
    warehouses.append(wh)
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', dtype=str, keep_default_na=False, index_col=0)
if set(trans_costs_df.columns) != set(customers):
    raise ValueError('transportation_costs.csv columns do not match customer list from demand.csv')
if set(trans_costs_df.index) != set(warehouses):
    raise ValueError('transportation_costs.csv rows do not match warehouse list from fixed_cost.csv')
cost = {}
for wh in warehouses:
    cost[wh] = {}
    for cust in customers:
        try:
            val = float(trans_costs_df.loc[wh, cust])
        except Exception:
            raise ValueError(f'Non-numeric transportation cost for warehouse {wh}, customer {cust}')
        cost[wh][cust] = val
M = sum((demand[cust] for cust in customers))
m = gp.Model('Bandcamp_UFLP')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[wh][cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)) + gp.quicksum((fixed_cost[wh] * y_vars[wh] for wh in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[wh, cust] for cust in customers)) <= M * y_vars[wh] for wh in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')