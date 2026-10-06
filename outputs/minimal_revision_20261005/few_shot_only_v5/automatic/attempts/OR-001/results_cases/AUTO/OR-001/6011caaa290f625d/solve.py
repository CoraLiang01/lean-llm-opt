import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv', ['customer', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv', ['Unnamed: 0', 'supply_capacity']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv', None)]

    def read_csv(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    df_demand = read_csv(csvs[0][0])
    if not set(['customer', 'demand']).issubset(df_demand.columns):
        raise ValueError('customer_demand.csv missing required columns.')
    customers = df_demand['customer'].astype(str).tolist()
    demand = df_demand.set_index('customer')['demand'].to_dict()
    df_supply = read_csv(csvs[1][0])
    if not set(['Unnamed: 0', 'supply_capacity']).issubset(df_supply.columns):
        raise ValueError('supply_capacity.csv missing required columns.')
    suppliers = df_supply['Unnamed: 0'].astype(str).tolist()
    supply_capacity = df_supply.set_index('Unnamed: 0')['supply_capacity'].to_dict()
    df_cost = read_csv(csvs[2][0])
    if df_cost.columns[0] != 'Unnamed: 0':
        raise ValueError('transportation_costs.csv first column must be supplier identifier.')
    cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customers)
    if missing_customers:
        raise ValueError(f'Customers {missing_customers} in demand not found in transportation_costs.csv.')
    cost_suppliers = df_cost['Unnamed: 0'].astype(str).tolist()
    missing_suppliers = set(suppliers) - set(cost_suppliers)
    if missing_suppliers:
        raise ValueError(f'Suppliers {missing_suppliers} in supply not found in transportation_costs.csv.')
    cost = {}
    for (_, row) in df_cost.iterrows():
        i = str(row['Unnamed: 0'])
        cost[i] = {}
        for j in customers:
            if j not in df_cost.columns:
                raise ValueError(f'Customer {j} not found in transportation_costs.csv columns.')
            cost[i][j] = row[j]
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in cost matrix.')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost for ({i},{j}) missing in cost matrix.')
    m = gp.Model('Amazon_Distribution')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()