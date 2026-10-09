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
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'monthly lot sizing data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    months = []
    demand = {}
    prod_cost = {}
    setup_cost = {}
    hold_cost = {}
    prod_cap = {}
    for (idx, row) in frame.iterrows():
        month = row['Month']
        months.append(month)
        try:
            demand[month] = float(row['Demand'])
            prod_cost[month] = float(row['ProductionCost'])
            setup_cost[month] = float(row['SetupCost'])
            hold_cost[month] = float(row['HoldingCost'])
            prod_cap[month] = float(row['ProductionCapacity'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in row {idx} for month {month}: {e}')
    if len(months) != 12:
        raise ValueError('Expected 12 months in data, got %d' % len(months))
    m = gp.Model('CapacitatedLotSizing')
    x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    inv_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + hold_cost[month] * inv_vars[month] for month in months)), gp.GRB.MINIMIZE)
    m.addConstr(x_vars[months[0]] == demand[months[0]] + inv_vars[months[0]], name='inv_balance_1')
    for t in range(1, len(months)):
        prev = months[t - 1]
        curr = months[t]
        m.addConstr(inv_vars[prev] + x_vars[curr] == demand[curr] + inv_vars[curr], name=f'inv_balance_{t + 1}')
    m.addConstr(inv_vars[months[-1]] == 0, name='ending_inventory')
    for month in months:
        m.addConstr(x_vars[month] <= prod_cap[month] * y_vars[month], name=f'capacity_link_{month}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')