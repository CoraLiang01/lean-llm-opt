import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    fixed_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    trans_df = pd.read_csv(transport_path, dtype=str, keep_default_na=False)
    demand_df['demand'] = demand_df['demand'].astype(float)
    fixed_df['fixed_costs'] = fixed_df['fixed_costs'].astype(float)
    for col in trans_df.columns[1:]:
        trans_df[col] = trans_df[col].astype(float)
    suppliers = list(fixed_df['Unnamed: 0'])
    customers = list(demand_df['customer'])
    fixed_cost = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
    demand = dict(zip(demand_df['customer'], demand_df['demand']))
    cost = {}
    for (_, row) in trans_df.iterrows():
        supplier = row['Unnamed: 0']
        cost[supplier] = {}
        for cust in customers:
            cost[supplier][cust] = row[cust]
    if set(suppliers) != set(fixed_cost.keys()):
        missing = set(fixed_cost.keys()) - set(suppliers)
        raise ValueError(f'Missing suppliers in fixed cost data: {missing}')
    if set(customers) != set(demand.keys()):
        missing = set(demand.keys()) - set(customers)
        raise ValueError(f'Missing customers in demand data: {missing}')
    for s in suppliers:
        if s not in cost:
            raise ValueError(f'Missing transportation costs for supplier {s}')
        if set(cost[s].keys()) != set(customers):
            missing = set(customers) - set(cost[s].keys())
            raise ValueError(f'Missing customer costs for supplier {s}: {missing}')
    M = sum(demand.values())
    model = gurobipy.Model('UFLP')
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    model.addConstrs((gurobipy.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()