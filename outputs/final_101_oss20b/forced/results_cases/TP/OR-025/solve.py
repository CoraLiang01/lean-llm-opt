CSVQA_DATA = {'query': 'A smartphone retail outlet offers several models with revenue data provided in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘TABLET’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for these products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘TABLET’ smartphone model i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SmartphoneRetailOutletSalesData.csv', 'records': [{'data_row_number': 454, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_10084.74', 'Revenue': '10084.74'}}, {'data_row_number': 455, 'values': {'Demand': '43', 'Initial Inventory': '300', 'Product Name': 'TABLET_12211.86', 'Revenue': '12211.86'}}, {'data_row_number': 456, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_14669.5', 'Revenue': '14669.5'}}, {'data_row_number': 457, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_14745.76', 'Revenue': '14745.76'}}, {'data_row_number': 458, 'values': {'Demand': '20', 'Initial Inventory': '100', 'Product Name': 'TABLET_14754.24', 'Revenue': '14754.24'}}, {'data_row_number': 459, 'values': {'Demand': '22', 'Initial Inventory': '110', 'Product Name': 'TABLET_16448.3', 'Revenue': '16448.3'}}, {'data_row_number': 460, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'TABLET_16448.31', 'Revenue': '16448.31'}}, {'data_row_number': 461, 'values': {'Demand': '16', 'Initial Inventory': '80', 'Product Name': 'TABLET_20143.22', 'Revenue': '20143.22'}}, {'data_row_number': 462, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_2042.38', 'Revenue': '2042.38'}}, {'data_row_number': 463, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_24915.25', 'Revenue': '24915.25'}}, {'data_row_number': 464, 'values': {'Demand': '32', 'Initial Inventory': '160', 'Product Name': 'TABLET_24915.26', 'Revenue': '24915.26'}}, {'data_row_number': 465, 'values': {'Demand': '14', 'Initial Inventory': '70', 'Product Name': 'TABLET_26448.3', 'Revenue': '26448.3'}}, {'data_row_number': 466, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_27042.38', 'Revenue': '27042.38'}}, {'data_row_number': 467, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_30000.0', 'Revenue': '30000.0'}}, {'data_row_number': 468, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_33397.46', 'Revenue': '33397.46'}}, {'data_row_number': 469, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_33398.3', 'Revenue': '33398.3'}}, {'data_row_number': 470, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_48567.8', 'Revenue': '48567.8'}}, {'data_row_number': 471, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'TABLET_48644.07', 'Revenue': '48644.07'}}, {'data_row_number': 472, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_50262.72', 'Revenue': '50262.72'}}, {'data_row_number': 473, 'values': {'Demand': '6', 'Initial Inventory': '30', 'Product Name': 'TABLET_53736.44', 'Revenue': '53736.44'}}, {'data_row_number': 474, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'TABLET_6957.62', 'Revenue': '6957.62'}}, {'data_row_number': 475, 'values': {'Demand': '12', 'Initial Inventory': '80', 'Product Name': 'TABLET_6957.63', 'Revenue': '6957.63'}}, {'data_row_number': 476, 'values': {'Demand': '60', 'Initial Inventory': '300', 'Product Name': 'TABLET_7550.84', 'Revenue': '7550.84'}}, {'data_row_number': 477, 'values': {'Demand': '8', 'Initial Inventory': '40', 'Product Name': 'TABLET_7550.85', 'Revenue': '7550.85'}}, {'data_row_number': 478, 'values': {'Demand': '8', 'Initial Inventory': '40', 'Product Name': 'TABLET_9584.74', 'Revenue': '9584.74'}}, {'data_row_number': 479, 'values': {'Demand': '38', 'Initial Inventory': '190', 'Product Name': 'TABLET_9661.02', 'Revenue': '9661.02'}}, {'data_row_number': 480, 'values': {'Demand': '4', 'Initial Inventory': '20', 'Product Name': 'TABLET_9669.5', 'Revenue': '9669.5'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == table_id:
            table = tbl
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError(f'No records found in table {table_id}')
    r_i = {}
    d_i = {}
    s_i = {}
    I = []
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if prod_name is None or rev_str is None or dem_str is None or (inv_str is None):
            raise ValueError(f"Missing required field in record {rec.get('data_row_number')}")
        try:
            revenue = float(rev_str)
            demand = int(dem_str)
            inventory = int(inv_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        if prod_name in r_i:
            raise ValueError(f'Duplicate product name {prod_name} in data')
        r_i[prod_name] = revenue
        d_i[prod_name] = demand
        s_i[prod_name] = inventory
        I.append(prod_name)
    m = gurobipy.Model('TabletRevenueMaximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d_i[i], name=f'demand_{i}')
        m.addConstr(x[i] <= s_i[i], name=f'inventory_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
