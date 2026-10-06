CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” '
          'while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost '
          'per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is '
          'to determine the optimal quantity of beverages to be shipped from each production plant to each retail '
          'outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while '
          'minimizing the total transportation cost.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'C1': '543.756480860856',
                                     'C2': '23.685276141764653',
                                     'C3': '23.676386730773032',
                                     'C4': '447.75143678673766',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '883.9151090405642',
                                     'C2': '0.04977684765576961',
                                     'C3': '0.0350986687216299',
                                     'C4': '44.45588531711622',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '537.3456896658107',
                                     'C2': '23.769274659075112',
                                     'C3': '498.95659249465467',
                                     'C4': '440.60737890439776',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1791.493192397229',
                                     'C2': '68.21633865655126',
                                     'C3': '1432.4837339656747',
                                     'C4': '1527.7635425462734',
                                     'Unnamed: 0': 'S4'}}],
             'returned_rows': 4,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    plants = [rec['values']['Unnamed: 0'] for rec in CSVQA_DATA['tables'][1]['records']]
    outlets = [rec['values']['customer'] for rec in CSVQA_DATA['tables'][0]['records']]
    demand = {rec['values']['customer']: float(rec['values']['demand']) for rec in CSVQA_DATA['tables'][0]['records']}
    supply_capacity = {rec['values']['Unnamed: 0']: float(rec['values']['supply_capacity']) for rec in CSVQA_DATA['tables'][1]['records']}
    cost = {}
    for rec in CSVQA_DATA['tables'][2]['records']:
        i = rec['values']['Unnamed: 0']
        cost[i] = {}
        for j in outlets:
            if j not in rec['values']:
                raise ValueError(f'Missing cost for plant {i} to outlet {j}')
            cost[i][j] = float(rec['values'][j])
    for i in plants:
        if i not in cost:
            raise ValueError(f'Missing cost row for plant {i}')
        for j in outlets:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for plant {i} to outlet {j}')
    for j in outlets:
        if j not in demand:
            raise ValueError(f'Missing demand for outlet {j}')
    for i in plants:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for plant {i}')
    m = gp.Model('BrewCo_TP')
    x = m.addVars(plants, outlets, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in outlets)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) >= demand[j] for j in outlets), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in outlets)) <= supply_capacity[i] for i in plants), name='')
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