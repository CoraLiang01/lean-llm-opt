CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 15 candidate plants F1–F15. Each plant can either be built (opened) or not. Building a plant '
          'incurs a fixed opening cost, and this cost is charged only once per plant, independent of how many '
          'customers the plant eventually serves. Once a plant is opened, it may supply products to any customers, '
          'subject to its own capacity.\n'
          '\n'
          '    There are 15 customers C1–C15 with fixed demands. If a plant is built, it incurs the fixed cost and can '
          'distribute to any number of customers. Shipping from plant i to customer j incurs a per-unit transport '
          'cost. The objective is to determine which plants should be built and how much to ship from each built plant '
          'to each customer in order to minimize the total cost, consisting of both fixed opening costs and variable '
          'transportation costs, while satisfying all customer demands and not exceeding the capacities of the opened '
          'plants.\n'
          '\n'
          '    The input data are given in two CSV files. cost.csv includes fixed_cost: opening cost of the plant '
          '(currency units; paid once if the plant is opened, regardless of the number of customers served). And '
          'capacity: plant capacity (maximum units the plant can supply). And also C1 … C15: per-unit transport cost '
          'from this plant to each customer. demand.csv includes demand of the customer (units) (C1-C15)',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['plant',
                         'fixed_cost',
                         'capacity',
                         'C1',
                         'C2',
                         'C3',
                         'C4',
                         'C5',
                         'C6',
                         'C7',
                         'C8',
                         'C9',
                         'C10',
                         'C11',
                         'C12',
                         'C13',
                         'C14',
                         'C15'],
             'file_index': 0,
             'file_name': 'cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0,
                          'values': {'C1': '7.8',
                                     'C10': '8.2',
                                     'C11': '7.3',
                                     'C12': '7.7',
                                     'C13': '6.7',
                                     'C14': '7.1',
                                     'C15': '7.9',
                                     'C2': '7.6',
                                     'C3': '6.7',
                                     'C4': '7.9',
                                     'C5': '8.1',
                                     'C6': '8.3',
                                     'C7': '7.3',
                                     'C8': '8.2',
                                     'C9': '8.1',
                                     'capacity': '101',
                                     'fixed_cost': '11250',
                                     'plant': 'F1'}},
                         {'source_row': 1,
                          'values': {'C1': '5.3',
                                     'C10': '6.1',
                                     'C11': '5',
                                     'C12': '5.6',
                                     'C13': '5.3',
                                     'C14': '4.9',
                                     'C15': '6.3',
                                     'C2': '6',
                                     'C3': '5',
                                     'C4': '6.4',
                                     'C5': '5.9',
                                     'C6': '6.2',
                                     'C7': '5.6',
                                     'C8': '6.1',
                                     'C9': '6.3',
                                     'capacity': '124',
                                     'fixed_cost': '13480',
                                     'plant': 'F2'}},
                         {'source_row': 2,
                          'values': {'C1': '7.2',
                                     'C10': '8.5',
                                     'C11': '7.2',
                                     'C12': '7.7',
                                     'C13': '7.1',
                                     'C14': '7.6',
                                     'C15': '8.4',
                                     'C2': '8.1',
                                     'C3': '7.4',
                                     'C4': '8.8',
                                     'C5': '8.5',
                                     'C6': '8.7',
                                     'C7': '7.7',
                                     'C8': '8.7',
                                     'C9': '8.9',
                                     'capacity': '139',
                                     'fixed_cost': '14870',
                                     'plant': 'F3'}},
                         {'source_row': 3,
                          'values': {'C1': '7',
                                     'C10': '7.3',
                                     'C11': '6.8',
                                     'C12': '7',
                                     'C13': '6.5',
                                     'C14': '6.7',
                                     'C15': '7.6',
                                     'C2': '7.1',
                                     'C3': '6.5',
                                     'C4': '7.9',
                                     'C5': '7.4',
                                     'C6': '7.7',
                                     'C7': '6.7',
                                     'C8': '7.9',
                                     'C9': '7.8',
                                     'capacity': '86',
                                     'fixed_cost': '10290',
                                     'plant': 'F4'}},
                         {'source_row': 4,
                          'values': {'C1': '3.5',
                                     'C10': '4',
                                     'C11': '3.2',
                                     'C12': '4',
                                     'C13': '2.9',
                                     'C14': '3.4',
                                     'C15': '3.9',
                                     'C2': '3.8',
                                     'C3': '2.9',
                                     'C4': '4.3',
                                     'C5': '3.6',
                                     'C6': '3.9',
                                     'C7': '3.2',
                                     'C8': '4.3',
                                     'C9': '4.5',
                                     'capacity': '157',
                                     'fixed_cost': '16740',
                                     'plant': 'F5'}},
                         {'source_row': 5,
                          'values': {'C1': '8.2',
                                     'C10': '9.2',
                                     'C11': '8.1',
                                     'C12': '8.7',
                                     'C13': '7.9',
                                     'C14': '8.5',
                                     'C15': '9',
                                     'C2': '8.6',
                                     'C3': '7.9',
                                     'C4': '9.5',
                                     'C5': '8.5',
                                     'C6': '9.3',
                                     'C7': '8.5',
                                     'C8': '9.4',
                                     'C9': '9',
                                     'capacity': '133',
                                     'fixed_cost': '13960',
                                     'plant': 'F6'}},
                         {'source_row': 6,
                          'values': {'C1': '6.9',
                                     'C10': '7.8',
                                     'C11': '6.9',
                                     'C12': '7.1',
                                     'C13': '7',
                                     'C14': '6.9',
                                     'C15': '7.5',
                                     'C2': '7.6',
                                     'C3': '6.8',
                                     'C4': '8.4',
                                     'C5': '8',
                                     'C6': '8',
                                     'C7': '7.6',
                                     'C8': '8',
                                     'C9': '8.1',
                                     'capacity': '118',
                                     'fixed_cost': '12680',
                                     'plant': 'F7'}},
                         {'source_row': 7,
                          'values': {'C1': '6.9',
                                     'C10': '7.9',
                                     'C11': '7',
                                     'C12': '7.4',
                                     'C13': '6.8',
                                     'C14': '7.3',
                                     'C15': '8',
                                     'C2': '7.8',
                                     'C3': '7.1',
                                     'C4': '8.7',
                                     'C5': '8.6',
                                     'C6': '8.2',
                                     'C7': '7.2',
                                     'C8': '7.9',
                                     'C9': '8.4',
                                     'capacity': '162',
                                     'fixed_cost': '17890',
                                     'plant': 'F8'}},
                         {'source_row': 8,
                          'values': {'C1': '3.5',
                                     'C10': '4.1',
                                     'C11': '3.2',
                                     'C12': '3.7',
                                     'C13': '3.7',
                                     'C14': '3.2',
                                     'C15': '4.5',
                                     'C2': '3.8',
                                     'C3': '2.8',
                                     'C4': '4.4',
                                     'C5': '4.2',
                                     'C6': '4.8',
                                     'C7': '3.8',
                                     'C8': '5',
                                     'C9': '4.5',
                                     'capacity': '92',
                                     'fixed_cost': '10950',
                                     'plant': 'F9'}},
                         {'source_row': 9,
                          'values': {'C1': '5.2',
                                     'C10': '5.9',
                                     'C11': '5.3',
                                     'C12': '6.1',
                                     'C13': '5.1',
                                     'C14': '5.2',
                                     'C15': '6.2',
                                     'C2': '6.1',
                                     'C3': '5.1',
                                     'C4': '6.3',
                                     'C5': '6.1',
                                     'C6': '6',
                                     'C7': '5.6',
                                     'C8': '6.5',
                                     'C9': '6.2',
                                     'capacity': '144',
                                     'fixed_cost': '15320',
                                     'plant': 'F10'}},
                         {'source_row': 10,
                          'values': {'C1': '5.2',
                                     'C10': '6.2',
                                     'C11': '5.2',
                                     'C12': '5.2',
                                     'C13': '4.5',
                                     'C14': '5.1',
                                     'C15': '5.4',
                                     'C2': '5.5',
                                     'C3': '4.5',
                                     'C4': '6.2',
                                     'C5': '5.7',
                                     'C6': '6.1',
                                     'C7': '5.1',
                                     'C8': '5.8',
                                     'C9': '5.7',
                                     'capacity': '107',
                                     'fixed_cost': '11830',
                                     'plant': 'F11'}},
                         {'source_row': 11,
                          'values': {'C1': '7.8',
                                     'C10': '8.4',
                                     'C11': '7.9',
                                     'C12': '8.2',
                                     'C13': '7.4',
                                     'C14': '7.6',
                                     'C15': '8.7',
                                     'C2': '8.7',
                                     'C3': '7.6',
                                     'C4': '9',
                                     'C5': '8.6',
                                     'C6': '9',
                                     'C7': '8.5',
                                     'C8': '9.3',
                                     'C9': '9.3',
                                     'capacity': '129',
                                     'fixed_cost': '14110',
                                     'plant': 'F12'}},
                         {'source_row': 12,
                          'values': {'C1': '6.7',
                                     'C10': '7.2',
                                     'C11': '6.3',
                                     'C12': '6.9',
                                     'C13': '6.2',
                                     'C14': '6',
                                     'C15': '7.2',
                                     'C2': '6.6',
                                     'C3': '6.1',
                                     'C4': '7.3',
                                     'C5': '7.1',
                                     'C6': '7.5',
                                     'C7': '6.7',
                                     'C8': '8',
                                     'C9': '7.6',
                                     'capacity': '151',
                                     'fixed_cost': '15970',
                                     'plant': 'F13'}},
                         {'source_row': 13,
                          'values': {'C1': '7.5',
                                     'C10': '8.1',
                                     'C11': '7.2',
                                     'C12': '7.3',
                                     'C13': '7',
                                     'C14': '7',
                                     'C15': '8',
                                     'C2': '8.6',
                                     'C3': '7.6',
                                     'C4': '8.2',
                                     'C5': '8',
                                     'C6': '7.9',
                                     'C7': '7.5',
                                     'C8': '8.7',
                                     'C9': '8.8',
                                     'capacity': '113',
                                     'fixed_cost': '13140',
                                     'plant': 'F14'}},
                         {'source_row': 14,
                          'values': {'C1': '5.1',
                                     'C10': '5.9',
                                     'C11': '5.1',
                                     'C12': '5.8',
                                     'C13': '5.4',
                                     'C14': '4.9',
                                     'C15': '6',
                                     'C2': '5.8',
                                     'C3': '4.6',
                                     'C4': '5.9',
                                     'C5': '6.5',
                                     'C6': '5.9',
                                     'C7': '5.2',
                                     'C8': '7',
                                     'C9': '7.1',
                                     'capacity': '85',
                                     'fixed_cost': '10580',
                                     'plant': 'F15'}}],
             'returned_rows': 15,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['customer', 'demand'],
             'file_index': 1,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '83'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '76'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '91'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '68'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '104'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '97'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '88'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '73'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '109'}},
                         {'source_row': 9, 'values': {'customer': 'C10', 'demand': '95'}},
                         {'source_row': 10, 'values': {'customer': 'C11', 'demand': '82'}},
                         {'source_row': 11, 'values': {'customer': 'C12', 'demand': '67'}},
                         {'source_row': 12, 'values': {'customer': 'C13', 'demand': '113'}},
                         {'source_row': 13, 'values': {'customer': 'C14', 'demand': '79'}},
                         {'source_row': 14, 'values': {'customer': 'C15', 'demand': '92'}}],
             'returned_rows': 15,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [15, 17], "
                                   "'expected_shape': [15, 15], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [15, 17], "
                                   "'expected_shape': [15, 15], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    cost_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            cost_table = t
            break
    if cost_table is None:
        raise RuntimeError('cost.csv table not found')
    plant_records = cost_table['records']
    plants = []
    fixed_cost = {}
    capacity = {}
    transport_cost = {}
    for rec in plant_records:
        v = rec['values']
        plant = v['plant']
        plants.append(plant)
        fixed_cost[plant] = float(v['fixed_cost'])
        capacity[plant] = float(v['capacity'])
        transport_cost[plant] = {}
        for cust_col in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']:
            transport_cost[plant][cust_col] = float(v[cust_col])
    demand_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            demand_table = t
            break
    if demand_table is None:
        raise RuntimeError('demand.csv table not found')
    customer_records = demand_table['records']
    customers = []
    demand = {}
    for rec in customer_records:
        v = rec['values']
        customer = v['customer']
        customers.append(customer)
        demand[customer] = float(v['demand'])
    for plant in plants:
        if plant not in fixed_cost or plant not in capacity or plant not in transport_cost:
            raise ValueError(f'Missing plant data for {plant}')
        for customer in customers:
            if customer not in transport_cost[plant]:
                raise ValueError(f'Missing transport cost for plant {plant}, customer {customer}')
    for customer in customers:
        if customer not in demand:
            raise ValueError(f'Missing demand for customer {customer}')
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in plants for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(plants, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()