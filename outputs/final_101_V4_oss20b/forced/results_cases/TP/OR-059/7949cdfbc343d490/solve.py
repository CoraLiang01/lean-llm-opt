CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of Colorado Motor Vehicle Sales, multiple car dealerships require inventory, and several '
          'suppliers located in different cities can provide the necessary vehicles. Each supplier incurs a fixed cost '
          'upon starting operations, with the fixed cost data provided in “fixed_cost.csv.” Each dealership needs to '
          'source a certain number of vehicles from these suppliers. For each dealership, the transportation cost per '
          'vehicle from each supplier is recorded in “transportation_costs.csv.” Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to open so that the demand of all dealerships "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of vehicles that '
          'dealership S_j sources from supplier F_i. For each dealership, x_{ij} represents the proportion of the '
          'total supply obtained from different suppliers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '4742532000'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '1600594000'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '5086889000'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '1027326000'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '11926044000'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '9058407000'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '5344367000'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '677201000'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '3236493000'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '100.64'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '98.72'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.18'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '96.58'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.75'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.06'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '101.78'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'fixed_costs': '93.86'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'C1': '1091.04',
                                     'C2': '85.72',
                                     'C3': '99.08',
                                     'C4': '747.35',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '58.88',
                                     'C2': '1617.16',
                                     'C3': '1786.44',
                                     'C4': '951.81',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '110.47', 'C2': '0.04', 'C3': '38.89', 'C4': '1397.95', 'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1458.85',
                                     'C2': '1049.27',
                                     'C3': '597.32',
                                     'C4': '1731.9',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '0.38',
                                     'C2': '2315.52',
                                     'C3': '1313.06',
                                     'C4': '1253.71',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '58.2', 'C2': '1395.81', 'C3': '84.6', 'C4': '830.64', 'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '1255.23',
                                     'C2': '1382.31',
                                     'C3': '78.79',
                                     'C4': '829.02',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '1990.09',
                                     'C2': '1.23',
                                     'C3': '38.97',
                                     'C4': '1396.35',
                                     'Unnamed: 0': 'S8'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    fixed_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        demand[cust] = float(rec['values']['demand'])
    fixed_cost = {}
    for rec in fixed_table['records']:
        sup = rec['values']['Unnamed: 0']
        fixed_cost[sup] = float(rec['values']['fixed_costs'])
    cost = {}
    for rec in cost_table['records']:
        sup = rec['values']['Unnamed: 0']
        cost[sup] = {}
        for cust in ['C1', 'C2', 'C3', 'C4']:
            cost[sup][cust] = float(rec['values'][cust])
    I = list(fixed_cost.keys())
    J = list(demand.keys())
    for sup in I:
        if sup not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        if sup not in cost:
            raise ValueError(f'Missing cost matrix for supplier {sup}')
        for cust in ['C1', 'C2', 'C3', 'C4']:
            if cust not in cost[sup]:
                raise ValueError(f'Missing cost for supplier {sup} to customer {cust}')
    for cust in J:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    total_demand = sum(demand.values())
    M_i = total_demand
    m = gurobipy.Model('Supplier_Opening_TP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I))
    for i in I:
        for j in cost[i]:
            obj += cost[i][j] * x[i, j]
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) >= demand[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= M_i * y[i], name=f'link_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()