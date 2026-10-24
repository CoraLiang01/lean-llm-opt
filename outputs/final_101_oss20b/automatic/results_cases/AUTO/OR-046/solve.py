CSVQA_DATA = {'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered each day.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '875'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity = None
    for table in CSVQA_DATA['tables']:
        if table['table_id'] == 'file_0_view_0':
            if not table['records']:
                raise ValueError('Capacity table has no records')
            record = table['records'][0]
            cap_val = record['values'].get('Capacity')
            if cap_val is None:
                raise ValueError('Capacity value missing')
            try:
                capacity = int(cap_val)
            except ValueError:
                raise ValueError(f'Capacity value not an integer: {cap_val}')
            break
    if capacity is None:
        raise ValueError('Capacity table not found')
    product_names = []
    value = {}
    weight = {}
    for table in CSVQA_DATA['tables']:
        if table['table_id'] == 'file_1_view_0':
            for record in table['records']:
                prod = record['values'].get('ProductName')
                if prod is None:
                    raise ValueError('ProductName missing in a record')
                val_str = record['values'].get('Value')
                if val_str is None:
                    raise ValueError(f'Value missing for product {prod}')
                w_str = record['values'].get('Weight')
                if w_str is None:
                    raise ValueError(f'Weight missing for product {prod}')
                try:
                    val_int = int(val_str)
                except ValueError:
                    raise ValueError(f'Value not an integer for product {prod}: {val_str}')
                try:
                    w_int = int(w_str)
                except ValueError:
                    raise ValueError(f'Weight not an integer for product {prod}: {w_str}')
                product_names.append(prod)
                value[prod] = val_int
                weight[prod] = w_int
            break
    if not product_names:
        raise ValueError('No product records found')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((value[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((weight[p] * x[p] for p in product_names)) <= capacity, name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
