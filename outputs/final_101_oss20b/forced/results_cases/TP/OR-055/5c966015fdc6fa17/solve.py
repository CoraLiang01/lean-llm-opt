CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A boat dealership needs to allocate different types of boats into different display areas. Specifically, '
          'the dealership has several display areas, each with a capacity limit provided in “capacity.csv.” The '
          'predefined value and size of each boat type can be found in “products.csv.” The objective is to determine '
          'the optimal number of units of each boat type to place in each display area to maximize the total value of '
          'the boats across all areas, while ensuring that the total size of the boats in each area does not exceed '
          'its capacity. The decision variables x_{ij} represent the number of units of boat type j to be placed in '
          'display area i.The decision variables must be integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['DisplayID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id_or_alias):
    for tbl in tables:
        if tbl['table_id'] == table_id_or_alias or table_id_or_alias in tbl.get('aliases', []):
            return tbl
    raise KeyError(f'Table {table_id_or_alias} not found')
cap_tbl = get_table('file_0_view_0')
capacities = {}
for rec in cap_tbl['records']:
    disp_id = rec['values']['DisplayID']
    cap = int(rec['values']['Capacity'])
    capacities[disp_id] = cap
prod_tbl = get_table('file_1_view_0')
products = []
for rec in prod_tbl['records']:
    name = rec['values']['ProductName']
    value = float(rec['values']['Value'])
    weight = float(rec['values']['Weight'])
    products.append({'name': name, 'value': value, 'weight': weight})
m = gp.Model('boat_allocation')
m.setParam(GRB.Param.MIPGap, 1e-06)
x = {}
for i in capacities:
    for p in products:
        var_name = f'x_{i}_{p['name'].replace(' ', '_')}'
        x[i, p['name']] = m.addVar(vtype=GRB.INTEGER, lb=0)
m.update()
for i, cap in capacities.items():
    expr = gp.quicksum((p['weight'] * x[i, p['name']] for p in products))
    m.addConstr(expr <= cap)
obj = gp.quicksum((p['value'] * x[i, p['name']] for i in capacities for p in products))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.status}')