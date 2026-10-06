CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are multiple suppliers responsible for delivering essential goods daily to customer groups located in '
          'different regions. Each supplier has a specific daily supply capacity, detailed in "supply_capacity.csv", '
          'while the daily demand of each customer group is recorded in "customer_demand.csv". The transportation cost '
          'per unit of goods from each supplier to each customer group is provided in "transportation_costs.csv". The '
          'objective is to determine the optimal transportation plan, specifying how goods should be allocated from '
          'each supplier to each customer group, ensuring that all demands are met without exceeding the supply '
          'capacity of any supplier, while minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'RA',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '216'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '168'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '264'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '216'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '216'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '192'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '144'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '168'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '168'}},
                         {'source_row': 9, 'values': {'customer': 'C10', 'demand': '168'}}],
             'returned_rows': 10,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '288'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '288'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '264'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '264'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '216'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '216'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '168'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '216'}},
                         {'source_row': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '240'}},
                         {'source_row': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '168'}}],
             'returned_rows': 10,
             'role': 'supplier capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'C1': '590.3648136504455',
                                     'C10': '58.895473920714416',
                                     'C2': '23.669172607322494',
                                     'C3': '88.89005869765714',
                                     'C4': '497.52228807074613',
                                     'C5': '466.09034321595647',
                                     'C6': '29.022096827063212',
                                     'C7': '23.675244833973835',
                                     'C8': '23.677760288117437',
                                     'C9': '0.3118394914937161',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '2042.0715001593626',
                                     'C10': '67.29170751036096',
                                     'C2': '2133.978484314172',
                                     'C3': '705.15912033561',
                                     'C4': '101.59454516295598',
                                     'C5': '2052.937657376311',
                                     'C6': '1738.754951414345',
                                     'C7': '101.61094965654742',
                                     'C8': '101.61062174376057',
                                     'C9': '122.45214268700826',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '22.297222160217984',
                                     'C10': '1008.6717394620979',
                                     'C2': '497.9271939314995',
                                     'C3': '1653.0828862073263',
                                     'C4': '23.68545123339267',
                                     'C5': '1386.0807887282344',
                                     'C6': '26.13715280763276',
                                     'C7': '497.6220482906461',
                                     'C8': '498.0935847133144',
                                     'C9': '865.3816296318804',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '960.7814533858373',
                                     'C10': '1351.8189071012546',
                                     'C2': '49.128300053752405',
                                     'C3': '1324.238697073691',
                                     'C4': '1032.2095478151716',
                                     'C5': '0.07804725392720868',
                                     'C6': '53.308268726049285',
                                     'C7': '49.1364167175093',
                                     'C8': '1031.8214424894484',
                                     'C9': '466.00495307991264',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1471.2721666392908',
                                     'C10': '1094.6695960752636',
                                     'C2': '85.6956072820555',
                                     'C3': '38.89266823851542',
                                     'C4': '1542.0500358120464',
                                     'C5': '112.20514372003504',
                                     'C6': '82.3702016356405',
                                     'C7': '1542.3399196620971',
                                     'C8': '85.69238745806277',
                                     'C9': '1924.9360769614245',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '191.9058726130392',
                                     'C10': '929.807168280051',
                                     'C2': '158.50401031820448',
                                     'C3': '91.02045349777458',
                                     'C4': '184.44747201726193',
                                     'C5': '968.146798696633',
                                     'C6': '284.1076062070199',
                                     'C7': '8.791061587686942',
                                     'C8': '158.70523835548545',
                                     'C9': '27.943874345249665',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '81.23891457326876',
                                     'C10': '849.9799406578097',
                                     'C2': '0.3744642223062507',
                                     'C3': '2079.46686537067',
                                     'C4': '0.3065671755503025',
                                     'C5': '1031.7772962191823',
                                     'C6': '7.203964492497209',
                                     'C7': '0.07623072241762692',
                                     'C8': '0.032473879548006554',
                                     'C9': '23.685827966421357',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '56.099310965461356',
                                     'C10': '1348.8366145919845',
                                     'C2': '935.6143108671334',
                                     'C3': '73.08824617002863',
                                     'C4': '52.00392409272077',
                                     'C5': '4.025792388934198',
                                     'C6': '1002.2327657984296',
                                     'C7': '935.7766029588662',
                                     'C8': '935.7007252277288',
                                     'C9': '612.8698719325438',
                                     'Unnamed: 0': 'S8'}},
                         {'source_row': 8,
                          'values': {'C1': '4.502283326860296',
                                     'C10': '40.46575554562011',
                                     'C2': '0.3899585342810754',
                                     'C3': '1782.4662178163346',
                                     'C4': '0.006345906612718274',
                                     'C5': '1031.9910114913148',
                                     'C6': '129.50665619510303',
                                     'C7': '0.2118319573481142',
                                     'C8': '0.645730107353115',
                                     'C9': '497.62723911435927',
                                     'Unnamed: 0': 'S9'}},
                         {'source_row': 9,
                          'values': {'C1': '333.6869270439132',
                                     'C10': '941.7526365563969',
                                     'C2': '277.4719386113677',
                                     'C3': '86.02096892455509',
                                     'C4': '277.30836609256806',
                                     'C5': '1004.4649084520337',
                                     'C6': '19.950336815857597',
                                     'C7': '13.202073690286834',
                                     'C8': '238.14321521805866',
                                     'C9': '411.0580332361589',
                                     'Unnamed: 0': 'S10'}}],
             'returned_rows': 10,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'expected_shape': [10, 10],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [10, 10]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    supply_table = tables['file_1_view_0']
    S = [rec['values']['Unnamed: 0'] for rec in supply_table['records']]
    a_s = {}
    for rec in supply_table['records']:
        s = rec['values']['Unnamed: 0']
        try:
            a_s[s] = float(rec['values']['supply_capacity'])
        except Exception:
            raise ValueError(f'Missing or invalid supply_capacity for supplier {s}')
    demand_table = tables['file_0_view_0']
    C = [rec['values']['customer'] for rec in demand_table['records']]
    d_c = {}
    for rec in demand_table['records']:
        c = rec['values']['customer']
        try:
            d_c[c] = float(rec['values']['demand'])
        except Exception:
            raise ValueError(f'Missing or invalid demand for customer {c}')
    cost_table = tables['file_2_view_0']
    t_sc = {}
    for rec in cost_table['records']:
        s = rec['values']['Unnamed: 0']
        t_sc[s] = {}
        for c in C:
            try:
                t_sc[s][c] = float(rec['values'][c])
            except Exception:
                raise ValueError(f'Missing or invalid transportation cost for ({s},{c})')
    if set(a_s.keys()) != set(S):
        raise ValueError('Mismatch in supplier identifiers between supply_capacity and cost matrix')
    if set(d_c.keys()) != set(C):
        raise ValueError('Mismatch in customer identifiers between customer_demand and cost matrix')
    for s in S:
        if set(t_sc[s].keys()) != set(C):
            raise ValueError(f'Cost matrix missing customers for supplier {s}')
    m = gp.Model('transportation')
    x = m.addVars(S, C, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((t_sc[s][c] * x[s, c] for s in S for c in C)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for c in C)) <= a_s[s] for s in S), name='')
    m.addConstrs((gp.quicksum((x[s, c] for s in S)) == d_c[c] for c in C), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)