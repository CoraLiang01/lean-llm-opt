CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A boat dealership needs to allocate different types of boats into different display areas. Specifically, '
          'the dealership has several display areas, each with a capacity limit provided in “capacity.csv.” The '
          'predefined value and size of each boat type can be found in “products.csv.” The objective is to determine '
          'the optimal number of units of each boat type to place in each display area to maximize the total value of '
          'the boats across all areas, while ensuring that the total size of the boats in each area does not exceed '
          'its capacity. The decision variables x_{ij} represent the number of units of boat type j to be placed in '
          'display area i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['DisplayID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'Capacity': '356', 'DisplayID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '478', 'DisplayID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '305', 'DisplayID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '291', 'DisplayID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '168', 'DisplayID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '449', 'DisplayID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '139', 'DisplayID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '383', 'DisplayID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '472', 'DisplayID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '288', 'DisplayID': '10'}},
                         {'source_row': 10, 'values': {'Capacity': '320', 'DisplayID': '11'}},
                         {'source_row': 11, 'values': {'Capacity': '250', 'DisplayID': '12'}},
                         {'source_row': 12, 'values': {'Capacity': '402', 'DisplayID': '13'}},
                         {'source_row': 13, 'values': {'Capacity': '293', 'DisplayID': '14'}}],
             'returned_rows': 14,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Speedboat', 'Value': '69978', 'Weight': '18'}},
                         {'source_row': 1, 'values': {'ProductName': 'Fishing Boat', 'Value': '54011', 'Weight': '42'}},
                         {'source_row': 2, 'values': {'ProductName': 'Catamaran', 'Value': '36352', 'Weight': '49'}},
                         {'source_row': 3, 'values': {'ProductName': 'Yacht', 'Value': '51521', 'Weight': '42'}},
                         {'source_row': 4, 'values': {'ProductName': 'Sailboat', 'Value': '50415', 'Weight': '41'}},
                         {'source_row': 5, 'values': {'ProductName': 'Kayak', 'Value': '76109', 'Weight': '48'}},
                         {'source_row': 6, 'values': {'ProductName': 'Canoe', 'Value': '50462', 'Weight': '22'}},
                         {'source_row': 7, 'values': {'ProductName': 'Houseboat', 'Value': '28989', 'Weight': '29'}},
                         {'source_row': 8, 'values': {'ProductName': 'Pontoon', 'Value': '23318', 'Weight': '45'}},
                         {'source_row': 9, 'values': {'ProductName': 'Jet Ski', 'Value': '26142', 'Weight': '14'}},
                         {'source_row': 10, 'values': {'ProductName': 'Rowboat', 'Value': '42040', 'Weight': '38'}},
                         {'source_row': 11, 'values': {'ProductName': 'Hovercraft', 'Value': '85961', 'Weight': '47'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'Cabin Cruiser', 'Value': '50142', 'Weight': '45'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Wakeboard Boat', 'Value': '48478', 'Weight': '28'}},
                         {'source_row': 14, 'values': {'ProductName': 'Dinghy', 'Value': '60953', 'Weight': '24'}},
                         {'source_row': 15, 'values': {'ProductName': 'Trawler', 'Value': '95265', 'Weight': '39'}},
                         {'source_row': 16, 'values': {'ProductName': 'Paddle Boat', 'Value': '22839', 'Weight': '32'}},
                         {'source_row': 17, 'values': {'ProductName': 'Submarine', 'Value': '90957', 'Weight': '36'}},
                         {'source_row': 18, 'values': {'ProductName': 'RIB', 'Value': '84652', 'Weight': '14'}},
                         {'source_row': 19, 'values': {'ProductName': 'Skiff', 'Value': '78991', 'Weight': '16'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    table_areas = None
    table_products = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table_areas = t
        elif t['table_id'] == 'file_1_view_0':
            table_products = t
    if table_areas is None or table_products is None:
        raise RuntimeError('Required tables not found in CSVQA_DATA.')
    I = []
    c = {}
    for rec in table_areas['records']:
        i = rec['values']['DisplayID']
        I.append(i)
        try:
            c[i] = int(rec['values']['Capacity'])
        except Exception:
            raise ValueError(f'Invalid capacity for DisplayID {i}')
    J = []
    v = {}
    w = {}
    for rec in table_products['records']:
        j = rec['values']['ProductName']
        J.append(j)
        try:
            v[j] = int(rec['values']['Value'])
            w[j] = int(rec['values']['Weight'])
        except Exception:
            raise ValueError(f'Invalid value/weight for ProductName {j}')
    if set(c.keys()) != set(I):
        raise ValueError('Missing capacities for some display areas.')
    if set(v.keys()) != set(J) or set(w.keys()) != set(J):
        raise ValueError('Missing value/weight for some products.')
    m = gp.Model('Boat_Display_Allocation')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in J)) <= c[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_DATA)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')