CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A coffee retail store needs to allocate some types of coffee products into different retail cabinets. '
          'Specifically, the store has some cabinets, each with a capacity limit provided in "capacity.csv." The '
          'predefined value and weight of each coffee product can be found in "products.csv." The objective is to '
          'determine the optimal number of units of each coffee product to place in each cabinet to maximize the total '
          'value of the products across all cabinets, while ensuring that the total weight of the coffee in each '
          'cabinet does not exceed its capacity. The decision variables x_ij represent the number of units of coffee '
          'product j to be placed in cabinet i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['CabinetID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'CabinetID': '1', 'Capacity': '400'}},
                         {'source_row': 1, 'values': {'CabinetID': '2', 'Capacity': '600'}},
                         {'source_row': 2, 'values': {'CabinetID': '3', 'Capacity': '500'}},
                         {'source_row': 3, 'values': {'CabinetID': '4', 'Capacity': '700'}},
                         {'source_row': 4, 'values': {'CabinetID': '5', 'Capacity': '450'}},
                         {'source_row': 5, 'values': {'CabinetID': '6', 'Capacity': '650'}},
                         {'source_row': 6, 'values': {'CabinetID': '7', 'Capacity': '550'}},
                         {'source_row': 7, 'values': {'CabinetID': '8', 'Capacity': '750'}},
                         {'source_row': 8, 'values': {'CabinetID': '9', 'Capacity': '480'}},
                         {'source_row': 9, 'values': {'CabinetID': '10', 'Capacity': '520'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 18,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}},
                         {'source_row': 2, 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}},
                         {'source_row': 3, 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}},
                         {'source_row': 5, 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}},
                         {'source_row': 7, 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}},
                         {'source_row': 9, 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}},
                         {'source_row': 12, 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}},
                         {'source_row': 13, 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}},
                         {'source_row': 15, 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}},
                         {'source_row': 17, 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}],
             'returned_rows': 18,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'CabinetID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'CabinetID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'CabinetID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'CabinetID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cabinets = []
    capacity = {}
    for rec in CSVQA_DATA['tables'][0]['records']:
        cab_id = rec['values']['CabinetID']
        cabinets.append(cab_id)
        capacity[cab_id] = float(rec['values']['Capacity'])
    products = []
    value = {}
    weight = {}
    for rec in CSVQA_DATA['tables'][1]['records']:
        prod = rec['values']['ProductName']
        products.append(prod)
        value[prod] = float(rec['values']['Value'])
        weight[prod] = float(rec['values']['Weight'])
    m = gp.Model()
    x = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in cabinets for j in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in cabinets), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()