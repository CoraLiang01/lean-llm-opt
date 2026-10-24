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
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        values = rec.get('values', {})
        required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
        for col in required_cols:
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('data_row_number')}")
        product = values['Product Name']
        if not product.startswith('S700_'):
            raise ValueError(f"Product name '{product}' does not start with 'S700_'")
        try:
            rev_val = float(values['Revenue'])
            dem_val = int(values['Demand'])
            inv_val = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Numeric conversion error in record {rec.get('data_row_number')}: {e}")
        if product in rev:
            raise ValueError(f"Duplicate product name '{product}' found")
        rev[product] = rev_val
        dem[product] = dem_val
        inv[product] = inv_val
    I = list(rev.keys())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= dem[i], name=f'dem_{i}')
        m.addConstr(x[i] <= inv[i], name=f'inv_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')
