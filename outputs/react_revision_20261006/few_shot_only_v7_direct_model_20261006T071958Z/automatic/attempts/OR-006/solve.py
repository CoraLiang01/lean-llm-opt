import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path, dtype=str, keep_default_na=False)
    supply_df = read_csv_robust(supply_path, dtype=str, keep_default_na=False)
    cost_df = read_csv_robust(cost_path, dtype=str, keep_default_na=False)
    warehouses = supply_df['Unnamed: 0'].tolist()
    stores = demand_df['customer'].tolist()
    try:
        demand = {}
        for (_, row) in demand_df.iterrows():
            key = row['customer']
            val = float(row['demand'])
            if key in demand:
                demand[key] += val
            else:
                demand[key] = val
    except Exception as e:
        raise RuntimeError(f'Error parsing demand: {e}')
    try:
        supply_capacity = {}
        for (_, row) in supply_df.iterrows():
            key = row['Unnamed: 0']
            val = float(row['supply_capacity'])
            if key in supply_capacity:
                supply_capacity[key] += val
            else:
                supply_capacity[key] = val
    except Exception as e:
        raise RuntimeError(f'Error parsing supply_capacity: {e}')
    try:
        cost = {}
        for (_, row) in cost_df.iterrows():
            i = row['Unnamed: 0']
            cost[i] = {}
            for j in stores:
                if j not in cost_df.columns:
                    raise RuntimeError(f"Store '{j}' not found as column in transportation_costs.csv")
                val = row[j]
                if val == '':
                    raise RuntimeError(f"Missing cost for warehouse '{i}', store '{j}'")
                cost[i][j] = float(val)
    except Exception as e:
        raise RuntimeError(f'Error parsing cost matrix: {e}')
    for i in warehouses:
        if i not in cost:
            raise RuntimeError(f"Warehouse '{i}' missing in cost matrix")
        for j in stores:
            if j not in cost[i]:
                raise RuntimeError(f"Cost missing for warehouse '{i}', store '{j}'")
    for j in stores:
        if j not in demand:
            raise RuntimeError(f"Demand missing for store '{j}'")
    for i in warehouses:
        if i not in supply_capacity:
            raise RuntimeError(f"Supply capacity missing for warehouse '{i}'")
    m = gp.Model('Logistics_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in warehouses for j in stores]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
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