CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A regional distributor can ship goods from depots to markets. Depot capacities are listed in '
          'depot_capacity.csv, market demands are listed in market_demand.csv, per-unit route costs are listed in '
          'route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A route may '
          'carry shipments only if it is activated.\n'
          '\n'
          'Formulate a minimum-cost fixed-charge transportation model. For each depot-market route i-j, define x_ij as '
          'the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is activated. The '
          'objective is to minimize variable shipping cost plus fixed route activation cost. The model should include '
          'market demand constraints, depot supply upper-bound constraints, shipment-to-route activation linking '
          'constraints using M_ij = min(depot capacity_i, market demand_j), nonnegativity constraints for shipment '
          'variables, and binary restrictions for route variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Depot', 'SupplyCapacity'],
             'file_index': 0,
             'file_name': 'depot_capacity.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Depot': 'D1', 'SupplyCapacity': '120'}},
                         {'source_row': 1, 'values': {'Depot': 'D2', 'SupplyCapacity': '100'}},
                         {'source_row': 2, 'values': {'Depot': 'D3', 'SupplyCapacity': '140'}},
                         {'source_row': 3, 'values': {'Depot': 'D4', 'SupplyCapacity': '90'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Market', 'Demand'],
             'file_index': 1,
             'file_name': 'market_demand.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Demand': '65', 'Market': 'M1'}},
                         {'source_row': 1, 'values': {'Demand': '80', 'Market': 'M2'}},
                         {'source_row': 2, 'values': {'Demand': '75', 'Market': 'M3'}},
                         {'source_row': 3, 'values': {'Demand': '95', 'Market': 'M4'}},
                         {'source_row': 4, 'values': {'Demand': '70', 'Market': 'M5'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4'],
             'file_index': 2,
             'file_name': 'route_variable_costs.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Depot': 'D1', 'M1': '4', 'M2': '5', 'M3': '17', 'M4': '18'}},
                         {'source_row': 1, 'values': {'Depot': 'D2', 'M1': '15', 'M2': '14', 'M3': '3', 'M4': '6'}},
                         {'source_row': 2, 'values': {'Depot': 'D3', 'M1': '18', 'M2': '16', 'M3': '15', 'M4': '4'}},
                         {'source_row': 3, 'values': {'Depot': 'D4', 'M1': '6', 'M2': '7', 'M3': '14', 'M4': '16'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'},
            {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4'],
             'file_index': 3,
             'file_name': 'route_fixed_costs.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'Depot': 'D1', 'M1': '240', 'M2': '270', 'M3': '560', 'M4': '590'}},
                         {'source_row': 1,
                          'values': {'Depot': 'D2', 'M1': '520', 'M2': '500', 'M3': '230', 'M4': '280'}},
                         {'source_row': 2,
                          'values': {'Depot': 'D3', 'M1': '610', 'M2': '580', 'M3': '540', 'M4': '260'}},
                         {'source_row': 3,
                          'values': {'Depot': 'D4', 'M1': '290', 'M2': '320', 'M3': '500', 'M4': '550'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_3_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
data = CSVQA_DATA

def solve_problem():

    def get_table(table_id):
        for tbl in data['tables']:
            if tbl['table_id'] == table_id:
                return tbl
        raise ValueError(f'Table {table_id} not found')
    depot_tbl = get_table('file_0_view_0')
    depots = []
    s = {}
    for rec in depot_tbl['records']:
        d = rec['values']['Depot']
        depots.append(d)
        s[d] = float(rec['values']['SupplyCapacity'])
    market_tbl = get_table('file_1_view_0')
    markets = []
    dmd = {}
    for rec in market_tbl['records']:
        m = rec['values']['Market']
        markets.append(m)
        dmd[m] = float(rec['values']['Demand'])
    varcost_tbl = get_table('file_2_view_0')
    c = {}
    for rec in varcost_tbl['records']:
        d = rec['values']['Depot']
        for m in markets:
            key = (d, m)
            if m not in rec['values']:
                raise ValueError(f'Missing variable cost for {key}')
            c[key] = float(rec['values'][m])
    fixed_tbl = get_table('file_3_view_0')
    f = {}
    for rec in fixed_tbl['records']:
        d = rec['values']['Depot']
        for m in markets:
            key = (d, m)
            if m not in rec['values']:
                raise ValueError(f'Missing fixed cost for {key}')
            f[key] = float(rec['values'][m])
    M = {}
    for d in depots:
        for m in markets:
            M[d, m] = min(s[d], dmd[m])
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(d, m) for d in depots for m in markets], vtype=GRB.CONTINUOUS, name='')
    y = m.addVars([(d, m) for d in depots for m in markets], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((c[d, m] * x[d, m] + f[d, m] * y[d, m] for d in depots for m in markets))
    m.setObjective(obj, GRB.MINIMIZE)
    for mkt in markets:
        m.addConstr(gurobipy.quicksum((x[d, mkt] for d in depots)) == dmd[mkt], name=f'Demand_{mkt}')
    for dpt in depots:
        m.addConstr(gurobipy.quicksum((x[dpt, m] for m in markets)) <= s[dpt], name=f'Supply_{dpt}')
    for dpt in depots:
        for mkt in markets:
            m.addConstr(x[dpt, mkt] <= M[dpt, mkt] * y[dpt, mkt], name=f'Link_{dpt}_{mkt}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()