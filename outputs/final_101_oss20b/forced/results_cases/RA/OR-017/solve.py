CSVQA_DATA = {'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'records': [{'data_row_number': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'data_row_number': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'data_row_number': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'data_row_number': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'data_row_number': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    p_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        values = rec['values']
        for field in ('SKU', 'Revenue', 'Demand', 'Initial Inventory'):
            if field not in values:
                raise ValueError(f"Missing field '{field}' in record {rec['data_row_number']}")
        sku = values['SKU']
        try:
            revenue = float(values['Revenue'])
            demand = int(values['Demand'])
            inventory = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec['data_row_number']}: {e}")
        I.append(sku)
        p_i[sku] = revenue
        d_i[sku] = demand
        s_i[sku] = inventory
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for i in I:
        m.addConstr(x[i] <= d_i[i], name=f'max_demand_{i}')
        m.addConstr(x[i] <= s_i[i], name=f'max_inventory_{i}')
    m.setObjective(gurobipy.quicksum((p_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
