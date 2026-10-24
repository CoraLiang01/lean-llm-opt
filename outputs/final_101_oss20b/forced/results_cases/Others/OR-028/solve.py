CSVQA_DATA = {'query': 'A company offers a variety of products with revenue data provided in the ‘Revenue’ column. The goal is to determine which demands to meet within the sales horizon using the initial inventory specified in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each product that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'WomenClothingEcommerceSalesData.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'sku_I27', 'Revenue': '238'}}, {'data_row_number': 1, 'values': {'Demand': '4', 'Initial Inventory': '20', 'Product Name': 'sku_I499', 'Revenue': '287'}}, {'data_row_number': 2, 'values': {'Demand': '16', 'Initial Inventory': '80', 'Product Name': 'sku_I719', 'Revenue': '268'}}, {'data_row_number': 3, 'values': {'Demand': '14', 'Initial Inventory': '70', 'Product Name': 'sku_T18', 'Revenue': '318'}}, {'data_row_number': 4, 'values': {'Demand': '4', 'Initial Inventory': '20', 'Product Name': 'sku_T29', 'Revenue': '207'}}, {'data_row_number': 5, 'values': {'Demand': '32', 'Initial Inventory': '160', 'Product Name': 'sku_T39', 'Revenue': '258'}}, {'data_row_number': 6, 'values': {'Demand': '8', 'Initial Inventory': '40', 'Product Name': 'sku_T499', 'Revenue': '249'}}, {'data_row_number': 7, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'sku_T9', 'Revenue': '227'}}, {'data_row_number': 8, 'values': {'Demand': '10', 'Initial Inventory': '50', 'Product Name': 'sku_3081', 'Revenue': '198'}}, {'data_row_number': 9, 'values': {'Demand': '8', 'Initial Inventory': '40', 'Product Name': 'sku_339', 'Revenue': '254'}}, {'data_row_number': 10, 'values': {'Demand': '18', 'Initial Inventory': '90', 'Product Name': 'sku_3799', 'Revenue': '246'}}, {'data_row_number': 11, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'sku_439', 'Revenue': '258'}}, {'data_row_number': 12, 'values': {'Demand': '4', 'Initial Inventory': '20', 'Product Name': 'sku_539', 'Revenue': '268'}}, {'data_row_number': 13, 'values': {'Demand': '8', 'Initial Inventory': '40', 'Product Name': 'sku_61399', 'Revenue': '278'}}, {'data_row_number': 14, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'sku_628', 'Revenue': '268'}}, {'data_row_number': 15, 'values': {'Demand': '198', 'Initial Inventory': '990', 'Product Name': 'sku_708', 'Revenue': '298'}}, {'data_row_number': 16, 'values': {'Demand': '32', 'Initial Inventory': '160', 'Product Name': 'sku_77', 'Revenue': '258'}}, {'data_row_number': 17, 'values': {'Demand': '18', 'Initial Inventory': '90', 'Product Name': 'sku_79', 'Revenue': '315'}}, {'data_row_number': 18, 'values': {'Demand': '570', 'Initial Inventory': '2870', 'Product Name': 'sku_799', 'Revenue': '264'}}, {'data_row_number': 19, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'sku_8499', 'Revenue': '238'}}, {'data_row_number': 20, 'values': {'Demand': '26', 'Initial Inventory': '130', 'Product Name': 'sku_89', 'Revenue': '258'}}, {'data_row_number': 21, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'sku_897', 'Revenue': '268'}}, {'data_row_number': 22, 'values': {'Demand': '33', 'Initial Inventory': '170', 'Product Name': 'sku_9699', 'Revenue': '288'}}, {'data_row_number': 23, 'values': {'Demand': '33', 'Initial Inventory': '170', 'Product Name': 'sku_bobo', 'Revenue': '228'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    ids = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        sid = rec.get('data_row_number')
        if sid is None:
            raise ValueError("Record missing 'data_row_number' key")
        ids.append(sid)
        vals = rec.get('values', {})
        try:
            rev = int(vals['Revenue'])
            dem = int(vals['Demand'])
            inv = int(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing required column in record {sid}: {e}')
        except ValueError as e:
            raise ValueError(f'Non-integer value in record {sid}: {e}')
        revenue[sid] = rev
        demand[sid] = dem
        inventory[sid] = inv
    for sid in ids:
        if sid not in revenue or sid not in demand or sid not in inventory:
            raise ValueError(f'Missing parameter for data_row_number {sid}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, lb=0, vtype=GRB.INTEGER, name='')
    obj = gurobipy.quicksum((revenue[s] * x[s] for s in ids))
    m.setObjective(obj, GRB.MAXIMIZE)
    for s in ids:
        m.addConstr(x[s] <= demand[s], name=f'demand_{s}')
        m.addConstr(x[s] <= inventory[s], name=f'inventory_{s}')
    m.optimize()
    return m
m = solve_problem()
