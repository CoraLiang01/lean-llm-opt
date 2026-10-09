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
             'role': 'monthly lot sizing parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem():
    frame = CSVQA_FRAMES['file_0_view_0']
    months = []
    demand = {}
    production_cost = {}
    setup_cost = {}
    holding_cost = {}
    production_capacity = {}
    month_to_t = {}
    for (idx, row) in frame.iterrows():
        month_label = row['Month']
        t = int(month_label[1:])
        months.append(t)
        month_to_t[month_label] = t
        demand[t] = float(row['Demand'])
        production_cost[t] = float(row['ProductionCost'])
        setup_cost[t] = float(row['SetupCost'])
        holding_cost[t] = float(row['HoldingCost'])
        production_capacity[t] = float(row['ProductionCapacity'])
    months = sorted(months)
    if set(months) != set(range(1, 13)):
        raise ValueError('Missing months in data: expected 1..12, got %s' % months)
    m = gp.Model('CapacitatedLotSizing')
    x_vars = m.addVars(months, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    inv_vars = m.addVars([0] + months, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((production_cost[t] * x_vars[t] + setup_cost[t] * y_vars[t] + holding_cost[t] * inv_vars[t] for t in months))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.addConstr(inv_vars[0] == 0, name='InitialInventory')
    for t in months:
        m.addConstr(inv_vars[t] == inv_vars[t - 1] + x_vars[t] - demand[t], name=f'InventoryBalance_{t}')
    for t in months:
        m.addConstr(x_vars[t] <= production_capacity[t] * y_vars[t], name=f'CapacityLink_{t}')
    m.addConstr(inv_vars[12] == 0, name='ZeroEndingInventory')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()