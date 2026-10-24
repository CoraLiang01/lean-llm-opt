CSVQA_DATA = {'query': 'An automobile sales company is planning its inventory replenishment strategy. For each type of vehicle, the company has a "products.csv" file that records the benefit coefficients for each vehicle type. The company also has an overall inventory capacity constraint, which is detailed in the "capacity.csv" file. The company\'s objective is to decide which vehicle types to order each day and in what quantities to maximize the overall benefit while adhering to the total inventory capacity. The decision variables x_i represent the number of units of vehicle type i to be ordered daily. The company needs to balance between maximizing benefits and complying with inventory constraints to develop the optimal ordering plan.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '1576'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedan', 'Value': '1752', 'Weight': '15'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUV', 'Value': '1856', 'Weight': '87'}}, {'data_row_number': 2, 'values': {'ProductName': 'Truck', 'Value': '8372', 'Weight': '36'}}, {'data_row_number': 3, 'values': {'ProductName': 'Convertible', 'Value': '6168', 'Weight': '30'}}, {'data_row_number': 4, 'values': {'ProductName': 'Minivan', 'Value': '9681', 'Weight': '33'}}, {'data_row_number': 5, 'values': {'ProductName': 'Coupe', 'Value': '8062', 'Weight': '72'}}, {'data_row_number': 6, 'values': {'ProductName': 'Hatchback', 'Value': '3895', 'Weight': '75'}}, {'data_row_number': 7, 'values': {'ProductName': 'Station Wagon', 'Value': '3254', 'Weight': '71'}}, {'data_row_number': 8, 'values': {'ProductName': 'Electric Car', 'Value': '1701', 'Weight': '51'}}, {'data_row_number': 9, 'values': {'ProductName': 'Hybrid Car', 'Value': '6799', 'Weight': '21'}}, {'data_row_number': 10, 'values': {'ProductName': 'Luxury Sedan', 'Value': '2724', 'Weight': '97'}}, {'data_row_number': 11, 'values': {'ProductName': 'Sports Car', 'Value': '6304', 'Weight': '52'}}, {'data_row_number': 12, 'values': {'ProductName': 'Crossover', 'Value': '3255', 'Weight': '25'}}, {'data_row_number': 13, 'values': {'ProductName': 'Diesel Truck', 'Value': '1923', 'Weight': '15'}}, {'data_row_number': 14, 'values': {'ProductName': 'Compact SUV', 'Value': '4103', 'Weight': '54'}}, {'data_row_number': 15, 'values': {'ProductName': 'Luxury SUV', 'Value': '4429', 'Weight': '57'}}, {'data_row_number': 16, 'values': {'ProductName': 'Cargo Van', 'Value': '2663', 'Weight': '18'}}, {'data_row_number': 17, 'values': {'ProductName': 'Pickup Truck', 'Value': '1691', 'Weight': '69'}}, {'data_row_number': 18, 'values': {'ProductName': 'Roadster', 'Value': '5632', 'Weight': '26'}}, {'data_row_number': 19, 'values': {'ProductName': 'Muscle Car', 'Value': '4793', 'Weight': '38'}}, {'data_row_number': 20, 'values': {'ProductName': 'Off-road Vehicle', 'Value': '1343', 'Weight': '31'}}, {'data_row_number': 21, 'values': {'ProductName': 'Camper Van', 'Value': '9124', 'Weight': '74'}}, {'data_row_number': 22, 'values': {'ProductName': 'Compact Car', 'Value': '3652', 'Weight': '82'}}, {'data_row_number': 23, 'values': {'ProductName': 'Motorcycle', 'Value': '8842', 'Weight': '49'}}, {'data_row_number': 24, 'values': {'ProductName': 'Electric SUV', 'Value': '9176', 'Weight': '64'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError('Capacity table not found')
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records')
    capacity_str = capacity_table['records'][0]['values']['Capacity']
    try:
        C = int(capacity_str)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {capacity_str}')
    products_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError('Products table not found')
    product_names = []
    c = {}
    w = {}
    for rec in products_table['records']:
        vals = rec['values']
        name = vals['ProductName']
        product_names.append(name)
        try:
            c_val = int(vals['Value'])
        except ValueError:
            raise ValueError(f'Invalid Value for product {name}')
        try:
            w_val = int(vals['Weight'])
        except ValueError:
            raise ValueError(f'Invalid Weight for product {name}')
        c[name] = c_val
        w[name] = w_val
    m = gurobipy.Model('VehicleInventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((c[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in product_names)) <= C)
    m.optimize()
    return m
m = solve_problem()
