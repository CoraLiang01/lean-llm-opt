CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A boat dealership needs to assign different types of boats (e.g. speedboats, fishing boats, catamarans, '
          'etc.) to different display areas. Specifically, the boat dealer has several display areas, and the capacity '
          'limit for each display area is provided in ‘capacity.csv’. Predefined values and dimensions for each vessel '
          'type can be found in ‘products.csv’. Our goal is to determine the optimal number of each vessel type to '
          'place in each showcase to maximise the total value of the vessels in all showcases, while ensuring that the '
          'total size of the vessels in each showcase does not exceed its capacity. The decision variable x_{ij} '
          'denotes the number of vessels of type j to be placed in display area i. The decision variable x_{ij} is the '
          'number of vessels of type j to be placed in display area i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['DisplayID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'Capacity': '457', 'DisplayID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '604', 'DisplayID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '751', 'DisplayID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '468', 'DisplayID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '343', 'DisplayID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '408', 'DisplayID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '741', 'DisplayID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '914', 'DisplayID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '682', 'DisplayID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '409', 'DisplayID': '10'}},
                         {'source_row': 10, 'values': {'Capacity': '342', 'DisplayID': '11'}},
                         {'source_row': 11, 'values': {'Capacity': '903', 'DisplayID': '12'}},
                         {'source_row': 12, 'values': {'Capacity': '680', 'DisplayID': '13'}},
                         {'source_row': 13, 'values': {'Capacity': '886', 'DisplayID': '14'}}],
             'returned_rows': 14,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Speedboat', 'Value': '29664', 'Weight': '18'}},
                         {'source_row': 1, 'values': {'ProductName': 'Fishing Boat', 'Value': '31778', 'Weight': '36'}},
                         {'source_row': 2, 'values': {'ProductName': 'Catamaran', 'Value': '73501', 'Weight': '25'}},
                         {'source_row': 3, 'values': {'ProductName': 'Yacht', 'Value': '78255', 'Weight': '16'}},
                         {'source_row': 4, 'values': {'ProductName': 'Sailboat', 'Value': '93606', 'Weight': '97'}},
                         {'source_row': 5, 'values': {'ProductName': 'Kayak', 'Value': '46983', 'Weight': '35'}},
                         {'source_row': 6, 'values': {'ProductName': 'Canoe', 'Value': '95026', 'Weight': '32'}},
                         {'source_row': 7, 'values': {'ProductName': 'Houseboat', 'Value': '57685', 'Weight': '100'}},
                         {'source_row': 8, 'values': {'ProductName': 'Pontoon', 'Value': '60323', 'Weight': '43'}},
                         {'source_row': 9, 'values': {'ProductName': 'Jet Ski', 'Value': '91224', 'Weight': '15'}},
                         {'source_row': 10, 'values': {'ProductName': 'Rowboat', 'Value': '44003', 'Weight': '95'}},
                         {'source_row': 11, 'values': {'ProductName': 'Hovercraft', 'Value': '75998', 'Weight': '57'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Cabin Cruiser', 'Value': '84525', 'Weight': '13'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Wakeboard Boat', 'Value': '66207', 'Weight': '44'}},
                         {'source_row': 14, 'values': {'ProductName': 'Dinghy', 'Value': '65002', 'Weight': '64'}},
                         {'source_row': 15, 'values': {'ProductName': 'Trawler', 'Value': '33132', 'Weight': '88'}},
                         {'source_row': 16, 'values': {'ProductName': 'Paddle Boat', 'Value': '69239', 'Weight': '42'}},
                         {'source_row': 17, 'values': {'ProductName': 'Submarine', 'Value': '66948', 'Weight': '46'}},
                         {'source_row': 18, 'values': {'ProductName': 'RIB', 'Value': '88240', 'Weight': '24'}},
                         {'source_row': 19, 'values': {'ProductName': 'Skiff', 'Value': '48858', 'Weight': '93'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'DisplayID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'DisplayID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'DisplayID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'DisplayID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = [record['values']['DisplayID'] for record in CSVQA_DATA['tables'][0]['records']]
    J = [record['values']['ProductName'] for record in CSVQA_DATA['tables'][1]['records']]
    c_i = {}
    for record in CSVQA_DATA['tables'][0]['records']:
        display_id = record['values']['DisplayID']
        capacity = record['values']['Capacity']
        c_i[display_id] = int(capacity)
    v_j = {}
    w_j = {}
    for record in CSVQA_DATA['tables'][1]['records']:
        product = record['values']['ProductName']
        value = record['values']['Value']
        weight = record['values']['Weight']
        v_j[product] = int(value)
        w_j[product] = int(weight)
    m = gp.Model('Boat_Display_Assignment')
    x = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * x[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()