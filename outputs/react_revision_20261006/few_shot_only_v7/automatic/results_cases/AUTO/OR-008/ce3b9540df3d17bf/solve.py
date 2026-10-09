import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'customer_demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv', 'supply_capacity': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv', 'transportation_costs': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    customer_demand_df = read_csv_with_encodings(csv_paths['customer_demand'])
    if 'Customers' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
        raise ValueError('customer_demand.csv missing required columns.')
    customers = customer_demand_df['Customers'].tolist()
    demand = {}
    for (_, row) in customer_demand_df.iterrows():
        cust = row['Customers']
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    supply_capacity_df = read_csv_with_encodings(csv_paths['supply_capacity'])
    if 'Suppliers' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
        raise ValueError('supply_capacity.csv missing required columns.')
    suppliers = supply_capacity_df['Suppliers'].tolist()
    supply_capacity = {}
    for (_, row) in supply_capacity_df.iterrows():
        sup = row['Suppliers']
        try:
            supply_capacity[sup] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
    transportation_costs_df = read_csv_with_encodings(csv_paths['transportation_costs'])
    if 'Unnamed: 0' not in transportation_costs_df.columns:
        raise ValueError('transportation_costs.csv missing supplier row identifier column.')
    cost = {}
    for (_, row) in transportation_costs_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            if cust not in transportation_costs_df.columns:
                raise ValueError(f'Customer {cust} missing in transportation_costs.csv columns.')
            try:
                cost_val = float(row[cust])
            except Exception:
                raise ValueError(f'Invalid cost value for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = cost_val
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Supplier {sup} missing in transportation_costs.csv.')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Cost for supplier {sup}, customer {cust} missing in transportation_costs.csv.')
    m = gp.Model('FreshMart_Transportation')
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
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