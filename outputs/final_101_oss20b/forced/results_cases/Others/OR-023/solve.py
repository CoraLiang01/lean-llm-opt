CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products with revenue data in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘ELE-S’. Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘ELE-S’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesStoreoverview.csv', 'records': [{'data_row_number': 36, 'values': {'Demand': '295', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10000463', 'Revenue': '4.0'}}, {'data_row_number': 37, 'values': {'Demand': '1002', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10000487', 'Revenue': '14.0'}}, {'data_row_number': 38, 'values': {'Demand': '958', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10003333', 'Revenue': '14.0'}}, {'data_row_number': 39, 'values': {'Demand': '777', 'Initial Inventory': '6000.0', 'Product_Reference': 'ELE-SMA-10009012', 'Revenue': '4.0'}}, {'data_row_number': 40, 'values': {'Demand': '271', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10009999', 'Revenue': '4.0'}}, {'data_row_number': 41, 'values': {'Demand': '244', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10011234', 'Revenue': '4.0'}}, {'data_row_number': 42, 'values': {'Demand': '990', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10027456', 'Revenue': '14.0'}}, {'data_row_number': 43, 'values': {'Demand': '1000', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10028567', 'Revenue': '14.0'}}, {'data_row_number': 44, 'values': {'Demand': '169', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10000484', 'Revenue': '2.4'}}, {'data_row_number': 45, 'values': {'Demand': '155', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10003030', 'Revenue': '2.4'}}, {'data_row_number': 46, 'values': {'Demand': '327', 'Initial Inventory': '2400.0', 'Product_Reference': 'ELE-SPE-10024123', 'Revenue': '2.4'}}, {'data_row_number': 47, 'values': {'Demand': '174', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10025234', 'Revenue': '2.4'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    records = table['records']
    I = [rec['data_row_number'] for rec in records]
    r = {}
    d = {}
    s = {}
    for rec in records:
        key = rec['data_row_number']
        vals = rec['values']
        try:
            r[key] = float(vals['Revenue'])
            d[key] = int(float(vals['Demand']))
            s[key] = float(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter {e} in record {key}')
    for key in I:
        if key not in r or key not in d or key not in s:
            raise ValueError(f'Missing data for key {key}')
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    obj = gurobipy.quicksum((r[i] * x[i] for i in I))
    m.setObjective(obj, GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'Demand_{i}')
        m.addConstr(x[i] <= s[i], name=f'Inventory_{i}')
        m.addConstr(x[i] >= 0, name=f'Nonneg_{i}')
    m.optimize()
    return m
m = solve_problem()
