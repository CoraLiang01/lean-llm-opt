CSVQA_DATA = {'query': 'A retail store offers a variety of best-selling products with profit data provided in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘S700_’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘S700_’ product i that the store intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SampleSalesData.csv', 'records': [{'data_row_number': 95, 'values': {'Demand': '1219', 'Initial Inventory': '9020', 'Product Name': 'S700_1138', 'Revenue': '70.67'}}, {'data_row_number': 96, 'values': {'Demand': '1127', 'Initial Inventory': '8370', 'Product Name': 'S700_1691', 'Revenue': '100.0'}}, {'data_row_number': 97, 'values': {'Demand': '1129', 'Initial Inventory': '8390', 'Product Name': 'S700_1938', 'Revenue': '70.15'}}, {'data_row_number': 98, 'values': {'Demand': '1176', 'Initial Inventory': '8680', 'Product Name': 'S700_2047', 'Revenue': '100.0'}}, {'data_row_number': 99, 'values': {'Demand': '1301', 'Initial Inventory': '9400', 'Product Name': 'S700_2466', 'Revenue': '100.0'}}, {'data_row_number': 100, 'values': {'Demand': '1340', 'Initial Inventory': '9900', 'Product Name': 'S700_2610', 'Revenue': '65.77'}}, {'data_row_number': 101, 'values': {'Demand': '1357', 'Initial Inventory': '9760', 'Product Name': 'S700_2824', 'Revenue': '100.0'}}, {'data_row_number': 102, 'values': {'Demand': '1158', 'Initial Inventory': '8610', 'Product Name': 'S700_2834', 'Revenue': '100.0'}}, {'data_row_number': 103, 'values': {'Demand': '1287', 'Initial Inventory': '9380', 'Product Name': 'S700_3167', 'Revenue': '74.4'}}, {'data_row_number': 104, 'values': {'Demand': '1281', 'Initial Inventory': '9170', 'Product Name': 'S700_3505', 'Revenue': '81.14'}}, {'data_row_number': 105, 'values': {'Demand': '1135', 'Initial Inventory': '8520', 'Product Name': 'S700_3962', 'Revenue': '100.0'}}, {'data_row_number': 106, 'values': {'Demand': '1392', 'Initial Inventory': '10290', 'Product Name': 'S700_4002', 'Revenue': '61.44'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    rev_dict = {}
    dem_dict = {}
    inv_dict = {}
    product_names = []
    for rec in records:
        vals = rec['values']
        prod = vals['Product Name']
        try:
            rev = float(vals['Revenue'])
            dem = float(vals['Demand'])
            inv = float(vals['Initial Inventory'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid numeric data for product {prod}') from e
        rev_dict[prod] = rev
        dem_dict[prod] = dem
        inv_dict[prod] = inv
        product_names.append(prod)
    for prod in product_names:
        if prod not in rev_dict:
            raise ValueError(f'Revenue missing for product {prod}')
        if prod not in dem_dict:
            raise ValueError(f'Demand missing for product {prod}')
        if prod not in inv_dict:
            raise ValueError(f'Inventory missing for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    model.addConstrs((x[i] <= dem_dict[i] for i in product_names), name='')
    model.addConstrs((x[i] <= inv_dict[i] for i in product_names), name='')
    model.setObjective(gurobipy.quicksum((rev_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
