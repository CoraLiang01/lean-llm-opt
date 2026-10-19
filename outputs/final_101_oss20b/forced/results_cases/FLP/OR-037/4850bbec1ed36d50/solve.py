CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car sales company is planning an inventory replenishment strategy. For each model (e.g., cars, SUVs, '
          'trucks, etc.), the company has a ‘products.csv’ file which records the profit that can be brought by '
          'selling each type of vehicle.The company has an overall inventory capacity limit, which is detailed in the '
          "‘capacity.csv’ file. The company's objective is to decide which types of vehicles to order each day and in "
          'what quantities, in order to maximise the overall benefit while adhering to the overall stock capacity. The '
          'decision variable x_i represents the number of vehicles of type i to be ordered per day. The company needs '
          'to strike a balance between maximising benefits and adhering to stock limits in order to develop an optimal '
          'ordering plan.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '765'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 25,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}},
                         {'source_row': 2, 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}},
                         {'source_row': 3, 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}},
                         {'source_row': 4, 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}},
                         {'source_row': 5, 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}},
                         {'source_row': 6, 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}},
                         {'source_row': 8, 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}},
                         {'source_row': 9, 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}},
                         {'source_row': 10, 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}},
                         {'source_row': 11, 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}},
                         {'source_row': 12, 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}},
                         {'source_row': 13, 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}},
                         {'source_row': 14, 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}},
                         {'source_row': 15, 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}},
                         {'source_row': 16, 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}},
                         {'source_row': 17, 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}},
                         {'source_row': 18, 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}},
                         {'source_row': 19, 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}},
                         {'source_row': 20,
                          'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}},
                         {'source_row': 21, 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}},
                         {'source_row': 22, 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}},
                         {'source_row': 23, 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}},
                         {'source_row': 24,
                          'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}],
             'returned_rows': 25,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
for table in CSVQA_DATA['tables']:
    if table.get('table_id') == 'file_0_view_0' or 'capacity.csv' in table.get('aliases', []):
        record = table['records'][0]
        capacity = int(record['values']['Capacity'])
    elif table.get('table_id') == 'file_1_view_0' or 'products.csv' in table.get('aliases', []):
        for rec in table['records']:
            pname = rec['values']['ProductName']
            value = float(rec['values']['Value'])
            weight = int(rec['values']['Weight'])
            products.append((pname, value, weight))
p = [prod[1] for prod in products]
w = [prod[2] for prod in products]
n = len(products)
m = gp.Model('Vehicle_Ordering')
x = m.addVars(n, lb=0)
m.setObjective(gp.quicksum((p[i] * x[i] for i in range(n))), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in range(n))) <= capacity)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective value:', m.ObjVal)
    for i in range(n):
        var = x[i]
        print(f'{var.VarName} = {var.X}')
else:
    print('Solver status:', m.Status)