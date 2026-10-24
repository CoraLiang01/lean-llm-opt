CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
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
    required_cols = {'Product_Name', 'Revenue', 'Demand', 'Initial Inventory'}
    r_i = {}
    d_i = {}
    s_i = {}
    I = []
    for rec in records:
        values = rec.get('values')
        if values is None:
            raise ValueError(f"Record {rec.get('data_row_number')} missing 'values' field")
        missing = required_cols - values.keys()
        if missing:
            raise ValueError(f"Record {rec.get('data_row_number')} missing columns: {missing}")
        prod_name = values['Product_Name']
        if prod_name in I:
            raise ValueError(f"Duplicate product name '{prod_name}' in records")
        I.append(prod_name)
        try:
            r_val = float(values['Revenue'])
            d_val = float(values['Demand'])
            s_val = float(values['Initial Inventory'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Non-numeric value in record {rec.get('data_row_number')}: {e}")
        r_i[prod_name] = r_val
        d_i[prod_name] = d_val
        s_i[prod_name] = s_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.addConstrs((x[i] <= d_i[i] for i in I), name='')
    m.addConstrs((x[i] <= s_i[i] for i in I), name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
