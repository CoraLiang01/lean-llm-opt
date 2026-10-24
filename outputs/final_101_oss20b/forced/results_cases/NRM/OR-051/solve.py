CSVQA_DATA = {'query': 'A coffee retail store needs to allocate some types of coffee products into different retail cabinets. Specifically, the store has some cabinets, each with a capacity limit provided in "capacity.csv." The predefined value and weight of each coffee product can be found in "products.csv." The objective is to determine the optimal number of units of each coffee product to place in each cabinet to maximize the total value of the products across all cabinets, while ensuring that the total weight of the coffee in each cabinet does not exceed its capacity. The decision variables x_ij represent the number of units of coffee product j to be placed in cabinet i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['CabinetID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'CabinetID': '1', 'Capacity': '400'}}, {'data_row_number': 1, 'values': {'CabinetID': '2', 'Capacity': '600'}}, {'data_row_number': 2, 'values': {'CabinetID': '3', 'Capacity': '500'}}, {'data_row_number': 3, 'values': {'CabinetID': '4', 'Capacity': '700'}}, {'data_row_number': 4, 'values': {'CabinetID': '5', 'Capacity': '450'}}, {'data_row_number': 5, 'values': {'CabinetID': '6', 'Capacity': '650'}}, {'data_row_number': 6, 'values': {'CabinetID': '7', 'Capacity': '550'}}, {'data_row_number': 7, 'values': {'CabinetID': '8', 'Capacity': '750'}}, {'data_row_number': 8, 'values': {'CabinetID': '9', 'Capacity': '480'}}, {'data_row_number': 9, 'values': {'CabinetID': '10', 'Capacity': '520'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}}, {'data_row_number': 1, 'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}}, {'data_row_number': 2, 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}}, {'data_row_number': 3, 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}}, {'data_row_number': 4, 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}}, {'data_row_number': 5, 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}}, {'data_row_number': 6, 'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}}, {'data_row_number': 7, 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}}, {'data_row_number': 8, 'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}}, {'data_row_number': 9, 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}}, {'data_row_number': 10, 'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}}, {'data_row_number': 11, 'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}}, {'data_row_number': 12, 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}}, {'data_row_number': 13, 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}}, {'data_row_number': 14, 'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}}, {'data_row_number': 15, 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}}, {'data_row_number': 16, 'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}}, {'data_row_number': 17, 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table_cabinets = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table_products = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = []
    capacity = {}
    for rec in table_cabinets['records']:
        vals = rec['values']
        if 'CabinetID' not in vals or 'Capacity' not in vals:
            raise ValueError(f"Missing required cabinet fields in record {rec['data_row_number']}")
        cid = vals['CabinetID']
        try:
            cap = int(vals['Capacity'])
        except ValueError:
            raise ValueError(f'Non-numeric capacity for cabinet {cid}')
        if cid in capacity:
            raise ValueError(f'Duplicate cabinet ID {cid}')
        I.append(cid)
        capacity[cid] = cap
    J = []
    value = {}
    weight = {}
    for rec in table_products['records']:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError(f"Missing required product fields in record {rec['data_row_number']}")
        pname = vals['ProductName']
        try:
            val = int(vals['Value'])
        except ValueError:
            raise ValueError(f'Non-numeric value for product {pname}')
        try:
            wt = float(vals['Weight'])
        except ValueError:
            raise ValueError(f'Non-numeric weight for product {pname}')
        if pname in value:
            raise ValueError(f'Duplicate product name {pname}')
        J.append(pname)
        value[pname] = val
        weight[pname] = wt
    if not I:
        raise ValueError('No cabinets found')
    if not J:
        raise ValueError('No products found')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((value[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i], name='cap_' + i)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')
