import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', 'supply': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', 'cost': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_robust(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Cannot decode {path} with tried encodings.')
    demand_df = read_csv_robust(csv_paths['demand'])
    supply_df = read_csv_robust(csv_paths['supply'])
    cost_df = read_csv_robust(csv_paths['cost'])
    if 'Unnamed: 0' in supply_df.columns:
        plants = supply_df['Unnamed: 0'].tolist()
    else:
        plants = supply_df.iloc[:, 0].tolist()
    if 'customer' in demand_df.columns:
        outlets = demand_df['customer'].tolist()
    else:
        outlets = demand_df.iloc[:, 0].tolist()
    if 'customer' in demand_df.columns and 'demand' in demand_df.columns:
        demand = {}
        for (_, row) in demand_df.iterrows():
            key = row['customer']
            try:
                val = float(row['demand'])
            except Exception:
                raise ValueError(f"Non-numeric demand for {key}: {row['demand']}")
            demand[key] = val
    else:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'.")
    supply_col = 'supply_capacity' if 'supply_capacity' in supply_df.columns else supply_df.columns[-1]
    supply = {}
    for (_, row) in supply_df.iterrows():
        key = row['Unnamed: 0'] if 'Unnamed: 0' in supply_df.columns else row.iloc[0]
        try:
            val = float(row[supply_col])
        except Exception:
            raise ValueError(f'Non-numeric supply for {key}: {row[supply_col]}')
        supply[key] = val
    if 'Unnamed: 0' in cost_df.columns:
        cost_df = cost_df.set_index('Unnamed: 0')
    else:
        cost_df = cost_df.set_index(cost_df.columns[0])
    cost = {}
    for i in plants:
        if i not in cost_df.index:
            raise ValueError(f'Plant {i} missing from transportation_costs.csv rows.')
        cost[i] = {}
        for j in outlets:
            if j not in cost_df.columns:
                raise ValueError(f'Outlet {j} missing from transportation_costs.csv columns.')
            try:
                cij = float(cost_df.loc[i, j])
            except Exception:
                raise ValueError(f'Non-numeric cost for ({i},{j}): {cost_df.loc[i, j]}')
            cost[i][j] = cij
    if set(demand.keys()) != set(outlets):
        raise ValueError('Mismatch between outlets in demand and outlets list.')
    if set(supply.keys()) != set(plants):
        raise ValueError('Mismatch between plants in supply and plants list.')
    for i in plants:
        for j in outlets:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for ({i},{j})')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in plants for j in outlets]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)), GRB.MINIMIZE)
    for j in outlets:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in plants)) >= demand[j], name=f'demand_{j}')
    for i in plants:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in outlets)) <= supply[i], name=f'supply_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()