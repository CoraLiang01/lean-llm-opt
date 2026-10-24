CSVQA_DATA = {'query': 'A retail store offers a variety of best-selling products with profit data provided in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘S700_’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘S700_’ product i that the store intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SampleSalesData.csv', 'records': [{'data_row_number': 95, 'values': {'Demand': '1219', 'Initial Inventory': '9020', 'Product Name': 'S700_1138', 'Revenue': '70.67'}}, {'data_row_number': 96, 'values': {'Demand': '1127', 'Initial Inventory': '8370', 'Product Name': 'S700_1691', 'Revenue': '100.0'}}, {'data_row_number': 97, 'values': {'Demand': '1129', 'Initial Inventory': '8390', 'Product Name': 'S700_1938', 'Revenue': '70.15'}}, {'data_row_number': 98, 'values': {'Demand': '1176', 'Initial Inventory': '8680', 'Product Name': 'S700_2047', 'Revenue': '100.0'}}, {'data_row_number': 99, 'values': {'Demand': '1301', 'Initial Inventory': '9400', 'Product Name': 'S700_2466', 'Revenue': '100.0'}}, {'data_row_number': 100, 'values': {'Demand': '1340', 'Initial Inventory': '9900', 'Product Name': 'S700_2610', 'Revenue': '65.77'}}, {'data_row_number': 101, 'values': {'Demand': '1357', 'Initial Inventory': '9760', 'Product Name': 'S700_2824', 'Revenue': '100.0'}}, {'data_row_number': 102, 'values': {'Demand': '1158', 'Initial Inventory': '8610', 'Product Name': 'S700_2834', 'Revenue': '100.0'}}, {'data_row_number': 103, 'values': {'Demand': '1287', 'Initial Inventory': '9380', 'Product Name': 'S700_3167', 'Revenue': '74.4'}}, {'data_row_number': 104, 'values': {'Demand': '1281', 'Initial Inventory': '9170', 'Product Name': 'S700_3505', 'Revenue': '81.14'}}, {'data_row_number': 105, 'values': {'Demand': '1135', 'Initial Inventory': '8520', 'Product Name': 'S700_3962', 'Revenue': '100.0'}}, {'data_row_number': 106, 'values': {'Demand': '1392', 'Initial Inventory': '10290', 'Product Name': 'S700_4002', 'Revenue': '61.44'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    revenue = {}
    demand = {}
    inventory = {}
    indices = []
    for record in target_table.get('records', []):
        values = record.get('values', {})
        for key in ('Product Name', 'Revenue', 'Demand', 'Initial Inventory'):
            if key not in values:
                raise ValueError(f"Missing field '{key}' in record {record.get('data_row_number')}")
        prod_name = values['Product Name']
        try:
            rev = float(values['Revenue'])
            dem = float(values['Demand'])
            inv = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Non-numeric value in record {record.get('data_row_number')}: {e}")
        indices.append(prod_name)
        revenue[prod_name] = rev
        demand[prod_name] = dem
        inventory[prod_name] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(indices, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    for i in indices:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inv_{i}')
    m.optimize()
    return m
m = solve_problem()
