import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path, dtype=str, keep_default_na=False)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'")
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
    supply_df = read_csv_with_encodings(supply_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in supply_df.columns or 'supply_capacity' not in supply_df.columns:
        raise ValueError("supply_capacity.csv must have columns 'Unnamed: 0' and 'supply_capacity'")
    suppliers = supply_df['Unnamed: 0'].tolist()
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        sup = row['Unnamed: 0']
        try:
            val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for supplier {sup}: {row['supply_capacity']}")
        if sup in supply_capacity:
            supply_capacity[sup] += val
        else:
            supply_capacity[sup] = val
    cost_df = read_csv_with_encodings(cost_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in cost_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'")
    for cust in customers:
        if cust not in cost_df.columns:
            raise ValueError(f'Customer {cust} missing in transportation_costs.csv columns')
    cost = {}
    for (_, row) in cost_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = val
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Missing cost row for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Missing cost for supplier {sup}, customer {cust}')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')