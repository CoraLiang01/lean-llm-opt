CSVQA_DATA = {'query': 'A retail establishment seeks to optimize the allocation of its merchandise across distinct categories (electronics, apparel, homeware, etc.) to achieve the highest possible total revenue. Each category maintains its own demand level, with revenue figures provided in the ‘Revenue’ column and current stock quantities documented in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. The optimization challenge involves determining the ideal fulfillment quantities x_i for each product that allocate available inventory while respecting inventory limits and maximizing total revenue.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailSalesDataset.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '240', 'Initial Inventory': '1570', 'Product Name': 'Beauty - 25', 'Revenue': '25'}}, {'data_row_number': 1, 'values': {'Demand': '202', 'Initial Inventory': '1330', 'Product Name': 'Beauty - 30', 'Revenue': '30'}}, {'data_row_number': 2, 'values': {'Demand': '216', 'Initial Inventory': '1420', 'Product Name': 'Beauty - 300', 'Revenue': '300'}}, {'data_row_number': 3, 'values': {'Demand': '263', 'Initial Inventory': '1700', 'Product Name': 'Beauty - 50', 'Revenue': '50'}}, {'data_row_number': 4, 'values': {'Demand': '256', 'Initial Inventory': '1690', 'Product Name': 'Beauty - 500', 'Revenue': '500'}}, {'data_row_number': 5, 'values': {'Demand': '281', 'Initial Inventory': '1840', 'Product Name': 'Clothing - 25', 'Revenue': '25'}}, {'data_row_number': 6, 'values': {'Demand': '261', 'Initial Inventory': '1710', 'Product Name': 'Clothing - 30', 'Revenue': '30'}}, {'data_row_number': 7, 'values': {'Demand': '295', 'Initial Inventory': '1930', 'Product Name': 'Clothing - 300', 'Revenue': '300'}}, {'data_row_number': 8, 'values': {'Demand': '290', 'Initial Inventory': '1890', 'Product Name': 'Clothing - 50', 'Revenue': '50'}}, {'data_row_number': 9, 'values': {'Demand': '244', 'Initial Inventory': '1570', 'Product Name': 'Clothing - 500', 'Revenue': '500'}}, {'data_row_number': 10, 'values': {'Demand': '273', 'Initial Inventory': '1810', 'Product Name': 'Electronics - 25', 'Revenue': '25'}}, {'data_row_number': 11, 'values': {'Demand': '220', 'Initial Inventory': '1410', 'Product Name': 'Electronics - 30', 'Revenue': '30'}}, {'data_row_number': 12, 'values': {'Demand': '286', 'Initial Inventory': '1830', 'Product Name': 'Electronics - 300', 'Revenue': '300'}}, {'data_row_number': 13, 'values': {'Demand': '268', 'Initial Inventory': '1750', 'Product Name': 'Electronics - 50', 'Revenue': '50'}}, {'data_row_number': 14, 'values': {'Demand': '262', 'Initial Inventory': '1690', 'Product Name': 'Electronics - 500', 'Revenue': '500'}}, {'data_row_number': 15, 'values': {'Demand': '255', 'Initial Inventory': '1660', 'Product Name': 'Home Goods - 25', 'Revenue': '25'}}, {'data_row_number': 16, 'values': {'Demand': '218', 'Initial Inventory': '1417', 'Product Name': 'Home Goods - 30', 'Revenue': '30'}}, {'data_row_number': 17, 'values': {'Demand': '278', 'Initial Inventory': '1807', 'Product Name': 'Home Goods - 50', 'Revenue': '50'}}, {'data_row_number': 18, 'values': {'Demand': '195', 'Initial Inventory': '1268', 'Product Name': 'Home Goods - 300', 'Revenue': '300'}}, {'data_row_number': 19, 'values': {'Demand': '248', 'Initial Inventory': '1612', 'Product Name': 'Home Goods - 500', 'Revenue': '500'}}, {'data_row_number': 20, 'values': {'Demand': '269', 'Initial Inventory': '1749', 'Product Name': 'Sports - 25', 'Revenue': '25'}}, {'data_row_number': 21, 'values': {'Demand': '227', 'Initial Inventory': '1476', 'Product Name': 'Sports - 30', 'Revenue': '30'}}, {'data_row_number': 22, 'values': {'Demand': '285', 'Initial Inventory': '1853', 'Product Name': 'Sports - 50', 'Revenue': '50'}}, {'data_row_number': 23, 'values': {'Demand': '208', 'Initial Inventory': '1352', 'Product Name': 'Sports - 300', 'Revenue': '300'}}, {'data_row_number': 24, 'values': {'Demand': '259', 'Initial Inventory': '1684', 'Product Name': 'Sports - 500', 'Revenue': '500'}}, {'data_row_number': 25, 'values': {'Demand': '242', 'Initial Inventory': '1573', 'Product Name': 'Furniture - 25', 'Revenue': '25'}}, {'data_row_number': 26, 'values': {'Demand': '213', 'Initial Inventory': '1385', 'Product Name': 'Furniture - 30', 'Revenue': '30'}}, {'data_row_number': 27, 'values': {'Demand': '272', 'Initial Inventory': '1768', 'Product Name': 'Furniture - 50', 'Revenue': '50'}}, {'data_row_number': 28, 'values': {'Demand': '224', 'Initial Inventory': '1456', 'Product Name': 'Furniture - 300', 'Revenue': '300'}}, {'data_row_number': 29, 'values': {'Demand': '251', 'Initial Inventory': '1632', 'Product Name': 'Furniture - 500', 'Revenue': '500'}}, {'data_row_number': 30, 'values': {'Demand': '257', 'Initial Inventory': '1671', 'Product Name': 'Toys - 25', 'Revenue': '25'}}, {'data_row_number': 31, 'values': {'Demand': '235', 'Initial Inventory': '1528', 'Product Name': 'Toys - 30', 'Revenue': '30'}}, {'data_row_number': 32, 'values': {'Demand': '291', 'Initial Inventory': '1892', 'Product Name': 'Toys - 50', 'Revenue': '50'}}, {'data_row_number': 33, 'values': {'Demand': '199', 'Initial Inventory': '1294', 'Product Name': 'Toys - 300', 'Revenue': '300'}}, {'data_row_number': 34, 'values': {'Demand': '264', 'Initial Inventory': '1716', 'Product Name': 'Toys - 500', 'Revenue': '500'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    target_table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == target_table_id:
            table = t
            break
    if table is None:
        raise ValueError(f"Table with id '{target_table_id}' not found in CSVQA_DATA")
    col_map = {'Product Name': 'I', 'Revenue': 'r_i', 'Demand': 'd_i', 'Initial Inventory': 's_i'}
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    I = []
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        values = rec.get('values', {})
        for col in col_map:
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('data_row_number')}")
        prod = values['Product Name']
        try:
            rev = float(values['Revenue'])
            dem = float(values['Demand'])
            inv = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        I.append(prod)
        r_i[prod] = rev
        d_i[prod] = dem
        s_i[prod] = inv
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= d_i[i], name=f'demand_{i}')
        model.addConstr(x[i] <= s_i[i], name=f'inventory_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
