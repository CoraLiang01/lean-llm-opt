CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    filtered = [rec for rec in records if '27in' in rec['values']['Product Name']]
    I = [rec['data_row_number'] for rec in filtered]
    r = {}
    d = {}
    s = {}
    for rec in filtered:
        i = rec['data_row_number']
        vals = rec['values']
        try:
            r[i] = float(vals['Revenue'])
            d[i] = int(vals['Demand'])
            s[i] = int(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in record {i}: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing data for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        m.addConstr(x[i] <= d[i], name='')
        m.addConstr(x[i] <= s[i], name='')
    obj = gurobipy.quicksum((r[i] * x[i] for i in I))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
