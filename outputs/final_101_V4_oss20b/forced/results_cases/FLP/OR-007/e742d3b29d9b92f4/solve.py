CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail chain, “GreenMart,” operates several warehouses that supply products to its various store '
          'locations. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply '
          'capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of '
          'product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to '
          'determine the optimal quantity of products to be shipped from each warehouse to each GreenMart store, '
          'ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while '
          'minimizing the total transportation cost.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'D1', 'demand': '428'}},
                         {'source_row': 1, 'values': {'customer': 'D2', 'demand': '217'}},
                         {'source_row': 2, 'values': {'customer': 'D3', 'demand': '214'}},
                         {'source_row': 3, 'values': {'customer': 'D4', 'demand': '380'}},
                         {'source_row': 4, 'values': {'customer': 'D5', 'demand': '254'}}],
             'returned_rows': 5,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['region', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'region': 'S1', 'supply_capacity': '428'}},
                         {'source_row': 1, 'values': {'region': 'S2', 'supply_capacity': '217'}},
                         {'source_row': 2, 'values': {'region': 'S3', 'supply_capacity': '214'}},
                         {'source_row': 3, 'values': {'region': 'S4', 'supply_capacity': '380'}},
                         {'source_row': 4, 'values': {'region': 'S5', 'supply_capacity': '254'}}],
             'returned_rows': 5,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'D1': '269.3910588020795',
                                     'D2': '1.4537335390933939',
                                     'D3': '99.60345345756605',
                                     'D4': '26.64078166309837',
                                     'D5': '9.537688956880922',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'D1': '9.291846876785183',
                                     'D2': '10.874778437070223',
                                     'D3': '144.52609291614627',
                                     'D4': '11.420133077898234',
                                     'D5': '153.1756819927813',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'D1': '9.674584301671008',
                                     'D2': '2.6191650959687944',
                                     'D3': '100.8242249168735',
                                     'D4': '3.2121910887916876',
                                     'D5': '133.8493396124168',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'D1': '270.57498480010247',
                                     'D2': '32.50253586',
                                     'D3': '4.6842098096469815',
                                     'D4': '1.5682269686546804',
                                     'D5': '9.58927599',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'D1': '226.0331910675782',
                                     'D2': '8.669161980826471',
                                     'D3': '65.47681316968448',
                                     'D4': '9.068765258459958',
                                     'D5': '202.65015316425533',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [5, 4], "
                                   "'expected_shape': [5, 5], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table('file_0_view_0')
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        val = float(rec['values']['demand'])
        demand[cust] = val
    supply_table = get_table('file_1_view_0')
    supply = {}
    for rec in supply_table['records']:
        reg = rec['values']['region']
        val = float(rec['values']['supply_capacity'])
        supply[reg] = val
    cost_table = get_table('file_2_view_0')
    cost_columns = [c for c in cost_table['columns'] if c != 'Unnamed: 0']
    cost = {}
    for rec in cost_table['records']:
        row_id = rec['values']['Unnamed: 0']
        for col in cost_columns:
            key = (row_id, col)
            cost[key] = float(rec['values'][col])
    I = list(supply.keys())
    J = list(demand.keys())
    missing_supply = set(I) - set(supply.keys())
    if missing_supply:
        raise ValueError(f'Missing supply data for warehouses: {missing_supply}')
    missing_demand = set(J) - set(demand.keys())
    if missing_demand:
        raise ValueError(f'Missing demand data for stores: {missing_demand}')
    missing_cost = [(i, j) for i in I for j in J if (i, j) not in cost]
    if missing_cost:
        raise ValueError(f'Missing cost data for pairs: {missing_cost}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= supply[i], name=f'supply_{i}')
    m.optimize()
    return m
m = solve_problem()