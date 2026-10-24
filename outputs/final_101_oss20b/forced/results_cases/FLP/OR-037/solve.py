CSVQA_DATA = {'query': "A car sales company is planning an inventory replenishment strategy. For each model (e.g., cars, SUVs, trucks, etc.), the company has a ‘products.csv’ file which records the profit that can be brought by selling each type of vehicle.The company has an overall inventory capacity limit, which is detailed in the ‘capacity.csv’ file. The company's objective is to decide which types of vehicles to order each day and in what quantities, in order to maximise the overall benefit while adhering to the overall stock capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day. The company needs to strike a balance between maximising benefits and adhering to stock limits in order to develop an optimal ordering plan.", 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '765'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'data_row_number': 2, 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'data_row_number': 3, 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'data_row_number': 4, 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'data_row_number': 5, 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'data_row_number': 6, 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'data_row_number': 7, 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'data_row_number': 8, 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'data_row_number': 9, 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'data_row_number': 10, 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'data_row_number': 11, 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'data_row_number': 12, 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'data_row_number': 13, 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'data_row_number': 14, 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'data_row_number': 15, 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'data_row_number': 16, 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'data_row_number': 17, 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'data_row_number': 18, 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'data_row_number': 19, 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'data_row_number': 20, 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'data_row_number': 21, 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'data_row_number': 22, 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'data_row_number': 23, 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'data_row_number': 24, 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity = None
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_0_view_0':
            if not table['records']:
                raise ValueError('Capacity table has no records')
            cap_str = table['records'][0]['values'].get('Capacity')
            if cap_str is None:
                raise ValueError('Capacity value missing')
            capacity = int(cap_str)
            break
    if capacity is None:
        raise ValueError('Capacity table not found')
    products = {}
    profit = {}
    weight = {}
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_1_view_0':
            for rec in table['records']:
                vals = rec['values']
                pname = vals.get('ProductName')
                if pname is None:
                    raise ValueError('ProductName missing in record')
                if pname in products:
                    raise ValueError(f'Duplicate product name: {pname}')
                prod_val_str = vals.get('Value')
                prod_wt_str = vals.get('Weight')
                if prod_val_str is None or prod_wt_str is None:
                    raise ValueError(f'Missing Value or Weight for product {pname}')
                try:
                    prod_val = int(prod_val_str)
                    prod_wt = int(prod_wt_str)
                except ValueError as e:
                    raise ValueError(f'Non-numeric Value/Weight for product {pname}') from e
                products[pname] = True
                profit[pname] = prod_val
                weight[pname] = prod_wt
            break
    if not products:
        raise ValueError('No products found')
    m = gurobipy.Model('InventoryReplenishment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products.keys(), lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((profit[i] * x[i] for i in products.keys())), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[i] * x[i] for i in products.keys())) <= capacity)
    m.optimize()
    return m
m = solve_problem()
