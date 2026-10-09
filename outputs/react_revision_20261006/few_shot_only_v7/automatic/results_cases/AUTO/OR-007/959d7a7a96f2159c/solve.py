import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv_with_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('customer_demand.csv must have columns: customer, demand')
    customers = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand:
            demand[cust] += val
        else:
            demand[cust] = val
    supply_df = read_csv_with_encodings(supply_path)
    if 'region' not in supply_df.columns or 'supply_capacity' not in supply_df.columns:
        raise ValueError('supply_capacity.csv must have columns: region, supply_capacity')
    warehouses = supply_df['region'].tolist()
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        wh = row['region']
        try:
            val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for warehouse {wh}: {row['supply_capacity']}")
        if wh in supply_capacity:
            supply_capacity[wh] += val
        else:
            supply_capacity[wh] = val
    cost_df = read_csv_with_encodings(cost_path)
    if cost_df.columns[0] not in ['Unnamed: 0', 'region', 'warehouse']:
        raise ValueError('First column of transportation_costs.csv must be warehouse identifier')
    cost_warehouses = cost_df[cost_df.columns[0]].tolist()
    cost_stores = [col for col in cost_df.columns[1:]]
    missing_wh = set(warehouses) - set(cost_warehouses)
    missing_cust = set(customers) - set(cost_stores)
    if missing_wh:
        raise ValueError(f'Warehouses missing in transportation_costs.csv: {missing_wh}')
    if missing_cust:
        raise ValueError(f'Customers missing in transportation_costs.csv: {missing_cust}')
    cost = {}
    for (idx, row) in cost_df.iterrows():
        wh = row[cost_df.columns[0]]
        cost[wh] = {}
        for cust in customers:
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric cost for warehouse {wh}, customer {cust}: {row[cust]}')
            cost[wh][cust] = val
    quantity_keys = [(wh, cust) for wh in warehouses for cust in customers]
    for (wh, cust) in quantity_keys:
        if wh not in cost or cust not in cost[wh]:
            raise ValueError(f'Missing cost coefficient for warehouse {wh}, customer {cust}')
    m = gp.Model('GreenMart_Transportation')
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[wh][cust] * quantity_vars[wh, cust] for (wh, cust) in quantity_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[wh, cust] for wh in warehouses)) >= demand[cust] for cust in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[wh, cust] for cust in customers)) <= supply_capacity[wh] for wh in warehouses), name='')
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