CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A manufacturing company plans production of a seasonal appliance over a 24-month horizon. The planning data '
          "are provided in monthly_lot_sizing.csv, including each month's demand, unit production cost, fixed setup "
          'cost, unit inventory holding cost, and ProductionCapacity. The ProductionCapacity column represents the '
          'maximum number of units that can be produced in that month; it is not an inventory storage capacity. '
          'Initial inventory is zero, unmet demand is not allowed, and the plan must end with zero inventory after '
          'month 24.\n'
          '\n'
          'Formulate a mixed-integer lot-sizing model. For each month t, define x_t as the production quantity, I_t as '
          'the ending inventory, and y_t as a binary variable equal to 1 if production is set up in month t and 0 '
          'otherwise. The objective is to minimize the total cost, including production costs, fixed setup costs, and '
          'inventory holding costs. The model should include monthly inventory-balance constraints, '
          'production-capacity/setup-linking constraints, the final zero-inventory requirement, nonnegativity '
          'constraints for production and inventory, and binary restrictions for the setup variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity'],
             'file_index': 0,
             'file_name': 'monthly_lot_sizing.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 24,
             'records': [{'source_row': 0,
                          'values': {'Demand': '80',
                                     'HoldingCost': '1.1',
                                     'Month': 'M01',
                                     'ProductionCapacity': '420',
                                     'ProductionCost': '18',
                                     'SetupCost': '900'}},
                         {'source_row': 1,
                          'values': {'Demand': '120',
                                     'HoldingCost': '1.2',
                                     'Month': 'M02',
                                     'ProductionCapacity': '360',
                                     'ProductionCost': '20',
                                     'SetupCost': '1100'}},
                         {'source_row': 2,
                          'values': {'Demand': '60',
                                     'HoldingCost': '1.0',
                                     'Month': 'M03',
                                     'ProductionCapacity': '380',
                                     'ProductionCost': '19',
                                     'SetupCost': '950'}},
                         {'source_row': 3,
                          'values': {'Demand': '150',
                                     'HoldingCost': '1.3',
                                     'Month': 'M04',
                                     'ProductionCapacity': '420',
                                     'ProductionCost': '21',
                                     'SetupCost': '1200'}},
                         {'source_row': 4,
                          'values': {'Demand': '90',
                                     'HoldingCost': '1.1',
                                     'Month': 'M05',
                                     'ProductionCapacity': '350',
                                     'ProductionCost': '18',
                                     'SetupCost': '850'}},
                         {'source_row': 5,
                          'values': {'Demand': '110',
                                     'HoldingCost': '1.4',
                                     'Month': 'M06',
                                     'ProductionCapacity': '430',
                                     'ProductionCost': '22',
                                     'SetupCost': '1300'}},
                         {'source_row': 6,
                          'values': {'Demand': '140',
                                     'HoldingCost': '1.2',
                                     'Month': 'M07',
                                     'ProductionCapacity': '390',
                                     'ProductionCost': '20',
                                     'SetupCost': '1000'}},
                         {'source_row': 7,
                          'values': {'Demand': '70',
                                     'HoldingCost': '1.0',
                                     'Month': 'M08',
                                     'ProductionCapacity': '360',
                                     'ProductionCost': '19',
                                     'SetupCost': '900'}},
                         {'source_row': 8,
                          'values': {'Demand': '160',
                                     'HoldingCost': '1.5',
                                     'Month': 'M09',
                                     'ProductionCapacity': '440',
                                     'ProductionCost': '23',
                                     'SetupCost': '1400'}},
                         {'source_row': 9,
                          'values': {'Demand': '100',
                                     'HoldingCost': '1.2',
                                     'Month': 'M10',
                                     'ProductionCapacity': '400',
                                     'ProductionCost': '21',
                                     'SetupCost': '1150'}},
                         {'source_row': 10,
                          'values': {'Demand': '130',
                                     'HoldingCost': '1.1',
                                     'Month': 'M11',
                                     'ProductionCapacity': '380',
                                     'ProductionCost': '20',
                                     'SetupCost': '1000'}},
                         {'source_row': 11,
                          'values': {'Demand': '85',
                                     'HoldingCost': '1.0',
                                     'Month': 'M12',
                                     'ProductionCapacity': '350',
                                     'ProductionCost': '18',
                                     'SetupCost': '850'}},
                         {'source_row': 12,
                          'values': {'Demand': '115',
                                     'HoldingCost': '1.1',
                                     'Month': 'M13',
                                     'ProductionCapacity': '390',
                                     'ProductionCost': '19',
                                     'SetupCost': '950'}},
                         {'source_row': 13,
                          'values': {'Demand': '95',
                                     'HoldingCost': '1.4',
                                     'Month': 'M14',
                                     'ProductionCapacity': '430',
                                     'ProductionCost': '22',
                                     'SetupCost': '1250'}},
                         {'source_row': 14,
                          'values': {'Demand': '145',
                                     'HoldingCost': '1.2',
                                     'Month': 'M15',
                                     'ProductionCapacity': '410',
                                     'ProductionCost': '21',
                                     'SetupCost': '1150'}},
                         {'source_row': 15,
                          'values': {'Demand': '105',
                                     'HoldingCost': '1.1',
                                     'Month': 'M16',
                                     'ProductionCapacity': '370',
                                     'ProductionCost': '20',
                                     'SetupCost': '1050'}},
                         {'source_row': 16,
                          'values': {'Demand': '75',
                                     'HoldingCost': '1.0',
                                     'Month': 'M17',
                                     'ProductionCapacity': '360',
                                     'ProductionCost': '18',
                                     'SetupCost': '800'}},
                         {'source_row': 17,
                          'values': {'Demand': '155',
                                     'HoldingCost': '1.5',
                                     'Month': 'M18',
                                     'ProductionCapacity': '440',
                                     'ProductionCost': '23',
                                     'SetupCost': '1350'}},
                         {'source_row': 18,
                          'values': {'Demand': '125',
                                     'HoldingCost': '1.2',
                                     'Month': 'M19',
                                     'ProductionCapacity': '400',
                                     'ProductionCost': '21',
                                     'SetupCost': '1100'}},
                         {'source_row': 19,
                          'values': {'Demand': '90',
                                     'HoldingCost': '1.1',
                                     'Month': 'M20',
                                     'ProductionCapacity': '360',
                                     'ProductionCost': '19',
                                     'SetupCost': '950'}},
                         {'source_row': 20,
                          'values': {'Demand': '135',
                                     'HoldingCost': '1.3',
                                     'Month': 'M21',
                                     'ProductionCapacity': '420',
                                     'ProductionCost': '22',
                                     'SetupCost': '1200'}},
                         {'source_row': 21,
                          'values': {'Demand': '65',
                                     'HoldingCost': '1.0',
                                     'Month': 'M22',
                                     'ProductionCapacity': '350',
                                     'ProductionCost': '18',
                                     'SetupCost': '850'}},
                         {'source_row': 22,
                          'values': {'Demand': '150',
                                     'HoldingCost': '1.2',
                                     'Month': 'M23',
                                     'ProductionCapacity': '430',
                                     'ProductionCost': '20',
                                     'SetupCost': '1000'}},
                         {'source_row': 23,
                          'values': {'Demand': '110',
                                     'HoldingCost': '0.0',
                                     'Month': 'M24',
                                     'ProductionCapacity': '390',
                                     'ProductionCost': '21',
                                     'SetupCost': '1100'}}],
             'returned_rows': 24,
             'role': 'monthly lot sizing planning data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    T = list(range(1, 25))
    d = {}
    c = {}
    f = {}
    h = {}
    u = {}
    for t in T:
        row = frame.iloc[t - 1]
        try:
            d[t] = float(row['Demand'])
            c[t] = float(row['ProductionCost'])
            f[t] = float(row['SetupCost'])
            h[t] = float(row['HoldingCost'])
            u[t] = float(row['ProductionCapacity'])
        except Exception as e:
            raise ValueError(f'Error parsing data for period t={t}: {e}')
    if not len(d) == len(c) == len(f) == len(h) == len(u) == 24:
        raise ValueError('Parameter dimension mismatch or missing data for some months.')
    m = gp.Model('MixedIntegerLotSizing')
    x_vars = m.addVars(T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    I_vars = m.addVars(T, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(T, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[t] * x_vars[t] + f[t] * y_vars[t] + h[t] * I_vars[t] for t in T)), gp.GRB.MINIMIZE)
    m.addConstr(x_vars[1] - d[1] == I_vars[1], name='inv_bal_1')
    for t in range(2, 25):
        m.addConstr(I_vars[t - 1] + x_vars[t] - d[t] == I_vars[t], name=f'inv_bal_{t}')
    m.addConstr(I_vars[24] == 0, name='final_inventory')
    for t in T:
        m.addConstr(x_vars[t] <= u[t] * y_vars[t], name=f'prod_cap_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)