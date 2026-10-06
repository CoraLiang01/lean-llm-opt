import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_try_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
    demand_df = read_csv_try_encodings(demand_path)
    fixed_cost_df = read_csv_try_encodings(fixed_cost_path)
    trans_cost_df = read_csv_try_encodings(trans_cost_path)
    suppliers = [str(i) for i in fixed_cost_df['Unnamed: 0']]
    dealerships = [str(j) for j in demand_df['customer']]
    demand = {}
    for (_, row) in demand_df.iterrows():
        j = str(row['customer'])
        if j in demand:
            demand[j] += row['demand']
        else:
            demand[j] = row['demand']
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        i = str(row['Unnamed: 0'])
        if i in fixed_cost:
            fixed_cost[i] += row['fixed_costs']
        else:
            fixed_cost[i] = row['fixed_costs']
    cost = {i: {} for i in suppliers}
    for (_, row) in trans_cost_df.iterrows():
        i = str(row['Unnamed: 0'])
        for j in dealerships:
            if j not in row:
                raise ValueError(f'Dealership {j} not found in transportation_costs.csv columns.')
            cost[i][j] = row[j]
    if set(demand.keys()) != set(dealerships):
        raise ValueError('Mismatch between demand dealerships and index set.')
    if set(fixed_cost.keys()) != set(suppliers):
        raise ValueError('Mismatch between fixed_cost suppliers and index set.')
    for i in suppliers:
        if set(cost[i].keys()) != set(dealerships):
            raise ValueError(f'Mismatch in cost matrix for supplier {i}.')
    M = sum((demand[j] for j in dealerships))
    m = gp.Model('ColoradoMotorVehicle_UFLP')
    x = m.addVars(suppliers, dealerships, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in dealerships)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in dealerships), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in dealerships)) <= M * y[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()