CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table_id = 'file_0_view_0'
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == table_id:
            table = tbl
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    I = []
    revenue = {}
    demand = {}
    inv = {}
    for record in table['records']:
        values = record['values']
        prod_name = values['Product Name']
        try:
            rev_val = float(values['Revenue'])
            dem_val = int(values['Demand'])
            inv_val = int(values['Initial Inventory'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid data for product {prod_name}') from e
        I.append(prod_name)
        revenue[prod_name] = rev_val
        demand[prod_name] = dem_val
        inv[prod_name] = inv_val
    for prod in I:
        if prod not in revenue or prod not in demand or prod not in inv:
            raise ValueError(f'Missing parameter for product {prod}')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'Demand_{i}')
    for i in I:
        m.addConstr(x[i] <= inv[i], name=f'Inv_{i}')
    m.optimize()
    return m
m = solve_problem()
