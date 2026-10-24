CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A snack distributor ships cartons from plants to regional retailers. Plant capacities are listed in '
          'plant_capacity.csv, retailer demands are listed in retailer_demand.csv, variable per-carton route costs are '
          'listed in route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A '
          'route can carry flow only if it is activated.\n'
          '\n'
          'Formulate a minimum-cost fixed-charge transportation model. For each plant-retailer route i-j, define x_ij '
          'as the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is opened. The '
          'objective is to minimize variable transportation cost plus fixed route-opening cost. The model should '
          'include retailer demand constraints, plant supply upper-bound constraints, shipment-to-route activation '
          'linking constraints using M_ij = min(plant capacity_i, retailer demand_j), nonnegativity constraints for '
          'shipment variables, and binary restrictions for route variables.',
 'relationships': [{'column_axis': {'id_column': 'Retailer', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Plant', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Plant',
                    'type': 'matrix'},
                   {'column_axis': {'id_column': 'Retailer', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_3_view_0',
                    'row_axis': {'id_column': 'Plant', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Plant',
                    'type': 'matrix'}],
 'route': 'Others',
 'tables': [{'columns': ['Plant', 'SupplyCapacity'],
             'file_index': 0,
             'file_name': 'plant_capacity.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Plant': 'P1', 'SupplyCapacity': '190'}},
                         {'source_row': 1, 'values': {'Plant': 'P2', 'SupplyCapacity': '160'}},
                         {'source_row': 2, 'values': {'Plant': 'P3', 'SupplyCapacity': '150'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Retailer', 'Demand'],
             'file_index': 1,
             'file_name': 'retailer_demand.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Demand': '70', 'Retailer': 'R1'}},
                         {'source_row': 1, 'values': {'Demand': '85', 'Retailer': 'R2'}},
                         {'source_row': 2, 'values': {'Demand': '75', 'Retailer': 'R3'}},
                         {'source_row': 3, 'values': {'Demand': '65', 'Retailer': 'R4'}},
                         {'source_row': 4, 'values': {'Demand': '95', 'Retailer': 'R5'}},
                         {'source_row': 5, 'values': {'Demand': '80', 'Retailer': 'R6'}}],
             'returned_rows': 6,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
             'file_index': 2,
             'file_name': 'route_variable_costs.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Plant': 'P1',
                                     'R1': '3',
                                     'R2': '4',
                                     'R3': '18',
                                     'R4': '20',
                                     'R5': '22',
                                     'R6': '19'}},
                         {'source_row': 1,
                          'values': {'Plant': 'P2',
                                     'R1': '17',
                                     'R2': '16',
                                     'R3': '4',
                                     'R4': '5',
                                     'R5': '20',
                                     'R6': '18'}},
                         {'source_row': 2,
                          'values': {'Plant': 'P3',
                                     'R1': '21',
                                     'R2': '19',
                                     'R3': '18',
                                     'R4': '17',
                                     'R5': '3',
                                     'R6': '4'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'},
            {'columns': ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
             'file_index': 3,
             'file_name': 'route_fixed_costs.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Plant': 'P1',
                                     'R1': '260',
                                     'R2': '300',
                                     'R3': '620',
                                     'R4': '670',
                                     'R5': '710',
                                     'R6': '660'}},
                         {'source_row': 1,
                          'values': {'Plant': 'P2',
                                     'R1': '610',
                                     'R2': '590',
                                     'R3': '280',
                                     'R4': '310',
                                     'R5': '690',
                                     'R6': '640'}},
                         {'source_row': 2,
                          'values': {'Plant': 'P3',
                                     'R1': '700',
                                     'R2': '650',
                                     'R3': '630',
                                     'R4': '600',
                                     'R5': '250',
                                     'R6': '290'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_3_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 6]},
                                  {'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 6],
                                   'matrix_table_id': 'file_3_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 6]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    plant_table = tables['file_0_view_0']
    s = {}
    for rec in plant_table['records']:
        plant = rec['values']['Plant']
        cap = float(rec['values']['SupplyCapacity'])
        s[plant] = cap
    retailer_table = tables['file_1_view_0']
    d = {}
    for rec in retailer_table['records']:
        retailer = rec['values']['Retailer']
        dem = float(rec['values']['Demand'])
        d[retailer] = dem
    var_cost_table = tables['file_2_view_0']
    c = {}
    for rec in var_cost_table['records']:
        plant = rec['values']['Plant']
        for retailer in d.keys():
            if retailer not in rec['values']:
                raise ValueError(f'Missing variable cost for plant {plant} retailer {retailer}')
            c[plant, retailer] = float(rec['values'][retailer])
    fixed_cost_table = tables['file_3_view_0']
    f = {}
    for rec in fixed_cost_table['records']:
        plant = rec['values']['Plant']
        for retailer in d.keys():
            if retailer not in rec['values']:
                raise ValueError(f'Missing fixed cost for plant {plant} retailer {retailer}')
            f[plant, retailer] = float(rec['values'][retailer])
    P = list(s.keys())
    R = list(d.keys())
    M = {(i, j): min(s[i], d[j]) for i in P for j in R}
    m = gurobipy.Model('fixed_charge_transport')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in P for j in R], lb=0.0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars([(i, j) for i in P for j in R], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in P for j in R))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in R:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in P)) == d[j], name=f'demand_{j}')
    for i in P:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in R)) <= s[i], name=f'supply_{i}')
    for i in P:
        for j in R:
            m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')