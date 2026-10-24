CSVQA_DATA = {'query': 'The company operates in the European market and offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'EuropeSalesRecords.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    view_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            view_table = tbl
            break
    if view_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = view_table.get('records', [])
    if not records:
        raise ValueError("No records found in table 'file_0_view_0'")
    I = []
    revenue = {}
    demand = {}
    init_inv = {}
    for rec in records:
        vals = rec.get('values', {})
        prod_name = vals.get('Product Name')
        if prod_name is None:
            raise ValueError(f"Record {rec.get('data_row_number')} missing 'Product Name'")
        I.append(prod_name)
        rev_str = vals.get('Revenue')
        dem_str = vals.get('Demand')
        init_str = vals.get('Initial Inventory')
        if rev_str is None or dem_str is None or init_str is None:
            raise ValueError(f"Record {rec.get('data_row_number')} missing required parameters")
        try:
            rev_val = float(rev_str)
            dem_val = int(dem_str)
            init_val = int(init_str)
        except ValueError as e:
            raise ValueError(f"Numeric conversion error in record {rec.get('data_row_number')}: {e}")
        revenue[prod_name] = rev_val
        demand[prod_name] = dem_val
        init_inv[prod_name] = init_val
    for i in I:
        if i not in revenue or i not in demand or i not in init_inv:
            raise ValueError(f'Missing parameter for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= demand[i], name='')
        model.addConstr(x[i] <= init_inv[i], name='')
    model.optimize()
    return model
m = solve_problem()
