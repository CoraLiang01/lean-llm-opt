import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    decode_attempts = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in decode_attempts:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    supply_df = read_csv_with_encodings(supply_path)
    cost_df = read_csv_with_encodings(cost_path)
    warehouses = supply_df['region'].tolist()
    stores = demand_df['customer'].tolist()
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in demand_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand: {e}')
    try:
        supply_capacity = {row['region']: float(row['supply_capacity']) for (_, row) in supply_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing supply_capacity: {e}')
    cost = {}
    cost_columns = [col for col in cost_df.columns if col != 'Unnamed: 0']
    for (_, row) in cost_df.iterrows():
        warehouse = row['Unnamed: 0']
        if warehouse not in warehouses:
            continue
        cost[warehouse] = {}
        for store in stores:
            if store in cost_columns:
                val = row[store]
            else:
                matches = [col for col in cost_columns if col.casefold() == store.casefold()]
                if matches:
                    val = row[matches[0]]
                else:
                    raise ValueError(f"Missing cost entry for warehouse '{warehouse}', store '{store}'")
            try:
                cost[warehouse][store] = float(val)
            except Exception as e:
                raise ValueError(f"Error parsing cost for warehouse '{warehouse}', store '{store}': {e}")
    for i in warehouses:
        if i not in cost:
            raise ValueError(f"Missing cost row for warehouse '{i}'")
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f"Missing cost entry for warehouse '{i}', store '{j}'")
    for j in stores:
        if j not in demand:
            raise ValueError(f"Missing demand for store '{j}'")
    for i in warehouses:
        if i not in supply_capacity:
            raise ValueError(f"Missing supply_capacity for warehouse '{i}'")
    keys = [(i, j) for i in warehouses for j in stores]
    m = gp.Model('GreenMart_Transportation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
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