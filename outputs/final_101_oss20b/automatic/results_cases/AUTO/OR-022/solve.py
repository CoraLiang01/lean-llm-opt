CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
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
    R = {}
    D = {}
    I = {}
    indices = []
    required_columns = {'Product Name', 'Revenue', 'Demand', 'Initial Inventory'}
    for record in table.get('records', []):
        values = record.get('values', {})
        if not required_columns.issubset(values.keys()):
            missing = required_columns - values.keys()
            raise ValueError(f"Missing columns {missing} in record {record.get('data_row_number')}")
        product_name = values['Product Name']
        try:
            revenue = float(values['Revenue'])
            demand = int(values['Demand'])
            inventory = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric conversion in record {record.get('data_row_number')}: {e}")
        if product_name in indices:
            raise ValueError(f'Duplicate product name {product_name} in data')
        indices.append(product_name)
        R[product_name] = revenue
        D[product_name] = demand
        I[product_name] = inventory
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(indices, lb=1, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((R[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    for i in indices:
        model.addConstr(x[i] <= D[i], name=f'demand_{i}')
        model.addConstr(x[i] <= I[i], name=f'inventory_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
