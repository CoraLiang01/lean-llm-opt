import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
plants = list(cost_df['Plant'])
customers = [col for col in cost_df.columns if col.startswith('C')]
try:
    fixed_cost = {row['Plant']: float(row['fixed_cost']) for _, row in cost_df.iterrows()}
    capacity = {row['Plant']: float(row['capacity']) for _, row in cost_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing fixed_cost or capacity: {e}')
cost = {}
for _, row in cost_df.iterrows():
    plant = row['Plant']
    cost[plant] = {}
    for cust in customers:
        try:
            cost[plant][cust] = float(row[cust])
        except Exception as e:
            raise ValueError(f'Error parsing cost for plant {plant}, customer {cust}: {e}')
if demand_df.shape[0] == 1:
    demand_row = demand_df.iloc[0]
    demand = {}
    for cust in customers:
        try:
            demand[cust] = float(demand_row[cust])
        except Exception as e:
            raise ValueError(f'Error parsing demand for customer {cust}: {e}')
else:
    raise ValueError('demand.csv must have exactly one row with customer demands.')
if set(plants) != set(fixed_cost.keys()) or set(plants) != set(capacity.keys()) or set(plants) != set(cost.keys()):
    raise ValueError('Mismatch in plant identifiers between cost.csv fields.')
if set(customers) != set(demand.keys()):
    raise ValueError('Mismatch in customer identifiers between cost.csv and demand.csv.')
for plant in plants:
    if set(cost[plant].keys()) != set(customers):
        raise ValueError(f'Plant {plant} missing cost entries for some customers.')
m = gp.Model('UFLP12')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(plants, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i] for i in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')