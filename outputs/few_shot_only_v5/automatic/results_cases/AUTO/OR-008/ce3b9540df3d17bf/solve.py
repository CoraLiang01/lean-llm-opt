import pandas as pd
import gurobipy as gp
from gurobipy import GRB
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv', dtype=str, keep_default_na=False, index_col=0)
customers = [idx.strip() for idx in customer_demand_df.index]
demand = {}
for cust in customers:
    val = customer_demand_df.loc[cust].values[0]
    try:
        demand[cust] = float(val)
    except Exception:
        raise ValueError(f'Invalid demand value for {cust}: {val}')
supply_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv', dtype=str, keep_default_na=False, index_col=0)
suppliers = [idx.strip() for idx in supply_capacity_df.index]
supply_capacity = {}
for sup in suppliers:
    val = supply_capacity_df.loc[sup].values[0]
    try:
        supply_capacity[sup] = float(val)
    except Exception:
        raise ValueError(f'Invalid supply capacity value for {sup}: {val}')
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', dtype=str, keep_default_na=False)
costs_df.rename(columns=lambda x: x.strip(), inplace=True)
costs_df['Unnamed: 0'] = costs_df['Unnamed: 0'].str.strip()
costs_df.set_index('Unnamed: 0', inplace=True)
if set(costs_df.index) != set(suppliers):
    raise ValueError('Mismatch between suppliers in supply_capacity.csv and transportation_costs.csv')
if set(costs_df.columns) != set(customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
cost = {}
for sup in suppliers:
    cost[sup] = {}
    for cust in customers:
        val = costs_df.loc[sup, cust]
        try:
            cost[sup][cust] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for {sup}, {cust}: {val}')
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sup, cust] for sup in suppliers)) >= demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[sup, cust] for cust in customers)) <= supply_capacity[sup] for sup in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')