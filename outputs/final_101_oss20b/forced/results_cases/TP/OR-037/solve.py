CSVQA_DATA = {'query': "A car sales company is planning an inventory replenishment strategy. For each model (e.g., cars, SUVs, trucks, etc.), the company has a ‘products.csv’ file which records the profit that can be brought by selling each type of vehicle.The company has an overall inventory capacity limit, which is detailed in the ‘capacity.csv’ file. The company's objective is to decide which types of vehicles to order each day and in what quantities, in order to maximise the overall benefit while adhering to the overall stock capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day. The company needs to strike a balance between maximising benefits and adhering to stock limits in order to develop an optimal ordering plan.", 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '765'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'data_row_number': 2, 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'data_row_number': 3, 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'data_row_number': 4, 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'data_row_number': 5, 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'data_row_number': 6, 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'data_row_number': 7, 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'data_row_number': 8, 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'data_row_number': 9, 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'data_row_number': 10, 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'data_row_number': 11, 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'data_row_number': 12, 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'data_row_number': 13, 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'data_row_number': 14, 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'data_row_number': 15, 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'data_row_number': 16, 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'data_row_number': 17, 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'data_row_number': 18, 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'data_row_number': 19, 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'data_row_number': 20, 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'data_row_number': 21, 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'data_row_number': 22, 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'data_row_number': 23, 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'data_row_number': 24, 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    capacity_table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA.")
    capacity_records = capacity_table.get('records', [])
    if not capacity_records:
        raise ValueError('No records found in capacity table.')
    capacity_value_str = capacity_records[0].get('values', {}).get('Capacity')
    if capacity_value_str is None:
        raise ValueError('Capacity value missing in capacity table.')
    try:
        capacity = float(capacity_value_str)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {capacity_value_str}')
    products_table = None
    for t in tables:
        if t.get('table_id') == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA.")
    v_dict = {}
    w_dict = {}
    index_set = []
    for record in products_table.get('records', []):
        values = record.get('values', {})
        product_name = values.get('ProductName')
        if product_name is None:
            raise ValueError('ProductName missing in a product record.')
        value_str = values.get('Value')
        weight_str = values.get('Weight')
        if value_str is None or weight_str is None:
            raise ValueError(f'Missing Value or Weight for product {product_name}.')
        try:
            v = float(value_str)
            w = float(weight_str)
        except ValueError:
            raise ValueError(f'Invalid numeric value for product {product_name}.')
        v_dict[product_name] = v
        w_dict[product_name] = w
        index_set.append(product_name)
    if not index_set:
        raise ValueError('No product records found in products table.')
    m = gurobipy.Model('InventoryReplenishment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(index_set, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((v_dict[i] * x[i] for i in index_set)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_dict[i] * x[i] for i in index_set)) <= capacity)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
