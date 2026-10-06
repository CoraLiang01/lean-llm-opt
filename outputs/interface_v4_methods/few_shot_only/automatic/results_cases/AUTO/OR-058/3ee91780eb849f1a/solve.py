CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. '
          'Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i '
          'are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent '
          'the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '216'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '216'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '144'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '144'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '144'}}],
             'returned_rows': 6,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}],
             'returned_rows': 6,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'C1': '0.08',
                                     'C2': '52.33',
                                     'C3': '73.56999999999999',
                                     'C4': '1237.33',
                                     'C5': '0.07000000000000001',
                                     'C6': '112.16',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '46.02',
                                     'C2': '175.23',
                                     'C3': '2026.83',
                                     'C4': '299.89',
                                     'C5': '966.53',
                                     'C6': '1590.42',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '1031.74',
                                     'C2': '78.13',
                                     'C3': '99.02',
                                     'C4': '277.07',
                                     'C5': '884.45',
                                     'C6': '1800.86',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '868.75',
                                     'C2': '94.2',
                                     'C3': '1776.34',
                                     'C4': '285.48',
                                     'C5': '868.85',
                                     'C6': '86.55',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1577',
                                     'C2': '760.15',
                                     'C3': '2090.19',
                                     'C4': '43.2',
                                     'C5': '1577.12',
                                     'C6': '1095.17',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '49.14',
                                     'C2': '4.33',
                                     'C3': '2079.57',
                                     'C4': '277.04',
                                     'C5': '1032.01',
                                     'C6': '1543.49',
                                     'Unnamed: 0': 'S6'}}],
             'returned_rows': 6,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_table = CSVQA_DATA['tables'][0]['records']
    customers = [rec['values']['customer'] for rec in demand_table]
    demand = {rec['values']['customer']: float(rec['values']['demand']) for rec in demand_table}
    fixed_table = CSVQA_DATA['tables'][1]['records']
    suppliers = [rec['values']['Unnamed: 0'] for rec in fixed_table]
    fixed_cost = {rec['values']['Unnamed: 0']: float(rec['values']['fixed_costs']) for rec in fixed_table}
    cost_table = CSVQA_DATA['tables'][2]['records']
    cost = {}
    for rec in cost_table:
        i = rec['values']['Unnamed: 0']
        cost[i] = {}
        for j in customers:
            cost[i][j] = float(rec['values'][j])
    M = sum((demand[j] for j in customers))
    if set(fixed_cost.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier identifiers between fixed_cost and suppliers list.')
    for i in suppliers:
        if set(cost[i].keys()) != set(customers):
            raise ValueError(f'Mismatch in customer identifiers for supplier {i} in cost matrix.')
    if set(demand.keys()) != set(customers):
        raise ValueError('Mismatch in customer identifiers between demand and customers list.')
    m = gp.Model('FLP')
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()