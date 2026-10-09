import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(np.int64)
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
transport_df['Unnamed: 0'] = transport_df['Unnamed: 0'].str.strip()
if set(suppliers) != set(transport_df['Unnamed: 0']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set([c for c in transport_df.columns if c != 'Unnamed: 0']):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_costs = {}
for (_, row) in transport_df.iterrows():
    i = row['Unnamed: 0']
    for j in customers:
        val = row[j]
        try:
            transport_costs[i, j] = float(val)
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {i}, customer {j}: {val}')
for i in suppliers:
    if i not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in customers:
        if (i, j) not in transport_costs:
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
M = sum(demand.values())

def solve_uflp(suppliers, customers, fixed_costs, transport_costs, demand, M):
    m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
    x_vars = m.addVars([(i, j) for i in suppliers for j in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_costs[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((x_vars[i, j] <= M * y_vars[i] for i in suppliers for j in customers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_uflp(suppliers, customers, fixed_costs, transport_costs, demand, M)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')