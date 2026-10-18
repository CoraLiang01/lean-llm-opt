import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer']
    try:
        demand[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost = {}
for idx, row in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0']
    try:
        fixed_cost[sup] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for supplier {sup}: {row['fixed_costs']}") from e
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
if list(trans_cost_df.columns)[0] != 'Unnamed: 0':
    raise ValueError("First column of transportation_costs.csv must be 'Unnamed: 0'")
cost = {}
for idx, row in trans_cost_df.iterrows():
    sup = row['Unnamed: 0']
    cost[sup] = {}
    for cust in customers:
        try:
            cost[sup][cust] = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}') from e
if set(suppliers) != set(cost.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
for sup in suppliers:
    if set(customers) != set(cost[sup].keys()):
        raise ValueError(f'Mismatch between customers in demand.csv and columns in transportation_costs.csv for supplier {sup}')
M = sum((demand[cust] for cust in customers))
m = gp.Model('Colorado_Motor_Vehicle_Sales_UFLP')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers)) + gp.quicksum((fixed_cost[sup] * y_vars[sup] for sup in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[sup, cust] for cust in customers)) <= M * y_vars[sup] for sup in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')