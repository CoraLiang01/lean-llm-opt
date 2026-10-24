CSVQA_DATA = {'query': 'A retail store offers a variety of best-selling products with profit data provided in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘S700_’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘S700_’ product i that the store intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SampleSalesData.csv', 'records': [{'data_row_number': 95, 'values': {'Demand': '1219', 'Initial Inventory': '9020', 'Product Name': 'S700_1138', 'Revenue': '70.67'}}, {'data_row_number': 96, 'values': {'Demand': '1127', 'Initial Inventory': '8370', 'Product Name': 'S700_1691', 'Revenue': '100.0'}}, {'data_row_number': 97, 'values': {'Demand': '1129', 'Initial Inventory': '8390', 'Product Name': 'S700_1938', 'Revenue': '70.15'}}, {'data_row_number': 98, 'values': {'Demand': '1176', 'Initial Inventory': '8680', 'Product Name': 'S700_2047', 'Revenue': '100.0'}}, {'data_row_number': 99, 'values': {'Demand': '1301', 'Initial Inventory': '9400', 'Product Name': 'S700_2466', 'Revenue': '100.0'}}, {'data_row_number': 100, 'values': {'Demand': '1340', 'Initial Inventory': '9900', 'Product Name': 'S700_2610', 'Revenue': '65.77'}}, {'data_row_number': 101, 'values': {'Demand': '1357', 'Initial Inventory': '9760', 'Product Name': 'S700_2824', 'Revenue': '100.0'}}, {'data_row_number': 102, 'values': {'Demand': '1158', 'Initial Inventory': '8610', 'Product Name': 'S700_2834', 'Revenue': '100.0'}}, {'data_row_number': 103, 'values': {'Demand': '1287', 'Initial Inventory': '9380', 'Product Name': 'S700_3167', 'Revenue': '74.4'}}, {'data_row_number': 104, 'values': {'Demand': '1281', 'Initial Inventory': '9170', 'Product Name': 'S700_3505', 'Revenue': '81.14'}}, {'data_row_number': 105, 'values': {'Demand': '1135', 'Initial Inventory': '8520', 'Product Name': 'S700_3962', 'Revenue': '100.0'}}, {'data_row_number': 106, 'values': {'Demand': '1392', 'Initial Inventory': '10290', 'Product Name': 'S700_4002', 'Revenue': '61.44'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    I = []
    r = {}
    d = {}
    v = {}
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('data_row_number')}")
        try:
            rev = float(values.get('Revenue', 0))
            dem = int(values.get('Demand', 0))
            inv = int(values.get('Initial Inventory', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        I.append(prod_name)
        r[prod_name] = rev
        d[prod_name] = dem
        v[prod_name] = inv
    for i in I:
        if i not in r or i not in d or i not in v:
            raise ValueError(f'Missing parameter for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'max_demand_{i}')
        m.addConstr(x[i] <= v[i], name=f'max_inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
