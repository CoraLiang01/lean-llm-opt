CSVQA_DATA = {'ignored_file_indices': [],
 'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to '
          'various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” '
          'while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost '
          'per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is '
          'to determine the optimal quantity of beverages to be shipped from each production plant to each retail '
          'outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while '
          'minimizing the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'Others',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '94'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '39'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '65'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '435'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [4, 4],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [4, 4]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise KeyError(f'Table {table_id} not found')
    cust_table = get_table('file_0_view_0')
    d_c = {}
    for rec in cust_table['records']:
        cust = rec['values']['customer']
        demand = float(rec['values']['demand'])
        d_c[cust] = demand
    C = list(d_c.keys())
    plant_table = get_table('file_1_view_0')
    s_p = {}
    for rec in plant_table['records']:
        plant = rec['values']['Unnamed: 0']
        capacity = float(rec['values']['supply_capacity'])
        s_p[plant] = capacity
    P = list(s_p.keys())
    cost_table = get_table('file_2_view_0')
    c_cost = {}
    for rec in cost_table['records']:
        plant = rec['values']['Unnamed: 0']
        for cust in C:
            cost = float(rec['values'][cust])
            c_cost[plant, cust] = cost
    for p in P:
        if p not in s_p:
            raise ValueError(f'Missing supply capacity for plant {p}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    for p in P:
        for c in C:
            if (p, c) not in c_cost:
                raise ValueError(f'Missing cost for plant {p} to customer {c}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(p, c) for p in P for c in C], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((c_cost[p, c] * x[p, c] for p in P for c in C)), GRB.MINIMIZE)
    for c in C:
        m.addConstr(gurobipy.quicksum((x[p, c] for p in P)) == d_c[c], name=f'demand_{c}')
    for p in P:
        m.addConstr(gurobipy.quicksum((x[p, c] for c in C)) <= s_p[p], name=f'capacity_{p}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()