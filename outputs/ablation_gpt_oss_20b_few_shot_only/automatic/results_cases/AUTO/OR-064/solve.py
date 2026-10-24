import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    df_fixed = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    df_transport = pd.read_csv(transport_path, dtype=str, keep_default_na=False)
    df_demand['demand'] = df_demand['demand'].astype(float)
    df_fixed['fixed_costs'] = df_fixed['fixed_costs'].astype(float)
    transport_numeric_cols = df_transport.columns[1:]
    df_transport[transport_numeric_cols] = df_transport[transport_numeric_cols].astype(float)
    df_demand.set_index('customer', inplace=True)
    df_fixed.set_index('Unnamed: 0', inplace=True)
    df_transport.set_index('Unnamed: 0', inplace=True)
    suppliers = list(df_fixed.index)
    customers = list(df_demand.index)
    if not set(suppliers).issubset(set(df_transport.index)):
        missing = set(suppliers) - set(df_transport.index)
        raise ValueError(f'Suppliers missing in transportation matrix: {missing}')
    if not set(customers).issubset(set(df_transport.columns)):
        missing = set(customers) - set(df_transport.columns)
        raise ValueError(f'Customers missing in transportation matrix columns: {missing}')
    f = df_fixed['fixed_costs'].to_dict()
    d = df_demand['demand'].to_dict()
    c = {(i, j): df_transport.at[i, j] for i in suppliers for j in customers}
    M = sum(d.values())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((c[i, j] * x[i, j] for i in suppliers for j in customers)) + gurobipy.quicksum((f[i] * y[i] for i in suppliers))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == d[j], name=f'demand_{j}')
    for i in suppliers:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in customers)) <= M * y[i], name=f'activation_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()