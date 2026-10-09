import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv', 'supply': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv', 'cost': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv(csv_paths['demand'])
    supply_df = read_csv(csv_paths['supply'])
    cost_df = read_csv(csv_paths['cost'])
    warehouses = []
    if 'Unnamed: 0' in supply_df.columns:
        warehouses = supply_df['Unnamed: 0'].tolist()
    else:
        warehouses = supply_df.iloc[:, 0].tolist()
    warehouses = [w for w in warehouses if w != '']
    stores = []
    if 'customer' in demand_df.columns:
        stores = demand_df['customer'].tolist()
    else:
        stores = demand_df.iloc[:, 0].tolist()
    stores = [s for s in stores if s != '']
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = row['customer'] if 'customer' in row else row.iloc[0]
        val = row['demand'] if 'demand' in row else row.iloc[1]
        try:
            demand[key] = float(val)
        except Exception:
            raise ValueError(f'Invalid demand value for {key}: {val}')
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        key = row['Unnamed: 0'] if 'Unnamed: 0' in row else row.iloc[0]
        val = row['supply_capacity'] if 'supply_capacity' in row else row.iloc[1]
        try:
            supply_capacity[key] = float(val)
        except Exception:
            raise ValueError(f'Invalid supply_capacity value for {key}: {val}')
    cost = {}
    if 'Unnamed: 0' in cost_df.columns:
        cost_df = cost_df.set_index('Unnamed: 0')
    else:
        cost_df = cost_df.set_index(cost_df.columns[0])
    for i in warehouses:
        if i not in cost_df.index:
            raise ValueError(f'Warehouse {i} missing in transportation_costs.csv')
        cost[i] = {}
        for j in stores:
            if j not in cost_df.columns:
                raise ValueError(f'Store {j} missing in transportation_costs.csv columns')
            val = cost_df.at[i, j]
            try:
                cost[i][j] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for ({i},{j}): {val}')
    for j in stores:
        if j not in demand:
            raise ValueError(f'Store {j} missing in demand data')
    for i in warehouses:
        if i not in supply_capacity:
            raise ValueError(f'Warehouse {i} missing in supply_capacity data')
    for i in warehouses:
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Cost for ({i},{j}) missing')
    m = gp.Model('Logistics_Transportation')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in warehouses for j in stores]
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()