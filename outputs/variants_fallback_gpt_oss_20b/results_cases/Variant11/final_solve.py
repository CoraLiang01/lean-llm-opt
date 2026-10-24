CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A manufacturer plans production of a seasonal spare part over 12 months. Monthly demand, unit production '
          'cost, setup cost, holding cost, and production capacity are listed in monthly_lot_sizing.csv. Initial '
          'inventory is zero, backlogging is not allowed, and ending inventory after the last month must be zero.\n'
          '\n'
          'Formulate a minimum-cost capacitated lot-sizing model. For each month t, define x_t as the nonnegative '
          'production quantity, inv_t as the nonnegative ending inventory, and y_t as a binary setup variable equal to '
          '1 if production occurs in month t. The objective is to minimize total production, setup, and holding cost. '
          'The model should include monthly inventory-balance constraints, production-capacity-to-setup linking '
          'constraints, the zero-ending-inventory constraint, nonnegativity constraints for production and inventory '
          'variables, and binary restrictions for setup variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity'],
             'file_index': 0,
             'file_name': 'monthly_lot_sizing.csv',
             'filters': {},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '95',
                                     'HoldingCost': '1.0',
                                     'Month': 'M01',
                                     'ProductionCapacity': '310',
                                     'ProductionCost': '17',
                                     'SetupCost': '760'}},
                         {'source_row': 1,
                          'values': {'Demand': '70',
                                     'HoldingCost': '1.1',
                                     'Month': 'M02',
                                     'ProductionCapacity': '260',
                                     'ProductionCost': '18',
                                     'SetupCost': '820'}},
                         {'source_row': 2,
                          'values': {'Demand': '130',
                                     'HoldingCost': '1.2',
                                     'Month': 'M03',
                                     'ProductionCapacity': '320',
                                     'ProductionCost': '20',
                                     'SetupCost': '930'}},
                         {'source_row': 3,
                          'values': {'Demand': '85',
                                     'HoldingCost': '1.0',
                                     'Month': 'M04',
                                     'ProductionCapacity': '300',
                                     'ProductionCost': '19',
                                     'SetupCost': '780'}},
                         {'source_row': 4,
                          'values': {'Demand': '115',
                                     'HoldingCost': '1.3',
                                     'Month': 'M05',
                                     'ProductionCapacity': '340',
                                     'ProductionCost': '21',
                                     'SetupCost': '960'}},
                         {'source_row': 5,
                          'values': {'Demand': '140',
                                     'HoldingCost': '1.4',
                                     'Month': 'M06',
                                     'ProductionCapacity': '360',
                                     'ProductionCost': '22',
                                     'SetupCost': '1040'}},
                         {'source_row': 6,
                          'values': {'Demand': '90',
                                     'HoldingCost': '1.1',
                                     'Month': 'M07',
                                     'ProductionCapacity': '290',
                                     'ProductionCost': '18',
                                     'SetupCost': '800'}},
                         {'source_row': 7,
                          'values': {'Demand': '125',
                                     'HoldingCost': '1.2',
                                     'Month': 'M08',
                                     'ProductionCapacity': '330',
                                     'ProductionCost': '20',
                                     'SetupCost': '900'}},
                         {'source_row': 8,
                          'values': {'Demand': '75',
                                     'HoldingCost': '1.0',
                                     'Month': 'M09',
                                     'ProductionCapacity': '280',
                                     'ProductionCost': '17',
                                     'SetupCost': '740'}},
                         {'source_row': 9,
                          'values': {'Demand': '150',
                                     'HoldingCost': '1.5',
                                     'Month': 'M10',
                                     'ProductionCapacity': '370',
                                     'ProductionCost': '23',
                                     'SetupCost': '1100'}},
                         {'source_row': 10,
                          'values': {'Demand': '105',
                                     'HoldingCost': '1.1',
                                     'Month': 'M11',
                                     'ProductionCapacity': '310',
                                     'ProductionCost': '19',
                                     'SetupCost': '850'}},
                         {'source_row': 11,
                          'values': {'Demand': '120',
                                     'HoldingCost': '0.0',
                                     'Month': 'M12',
                                     'ProductionCapacity': '350',
                                     'ProductionCost': '21',
                                     'SetupCost': '980'}}],
             'returned_rows': 12,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    source_rows = sorted((rec['source_row'] for rec in records))
    D = {}
    c = {}
    s = {}
    h = {}
    C = {}
    for rec in records:
        r = rec['source_row']
        vals = rec['values']
        D[r] = float(vals['Demand'])
        c[r] = float(vals['ProductionCost'])
        s[r] = float(vals['SetupCost'])
        h[r] = float(vals['HoldingCost'])
        C[r] = float(vals['ProductionCapacity'])
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(source_rows, lb=0, vtype=GRB.CONTINUOUS, name='')
    inv = m.addVars(source_rows, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(source_rows, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c[r] * x[r] + s[r] * y[r] + h[r] * inv[r] for r in source_rows)), GRB.MINIMIZE)
    for r in source_rows:
        if r == 0:
            m.addConstr(x[r] == D[r] + inv[r], name=f'invbal_{r}')
        else:
            prev = r - 1
            m.addConstr(inv[prev] + x[r] == D[r] + inv[r], name=f'invbal_{r}')
        m.addConstr(x[r] <= C[r] * y[r], name=f'cap_{r}')
    last = source_rows[-1]
    m.addConstr(inv[last] == 0, name='zero_end_inv')
    m.optimize()
    return m
m = solve_problem()