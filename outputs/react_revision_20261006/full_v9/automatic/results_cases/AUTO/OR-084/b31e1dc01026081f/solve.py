CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Process',
                         '1',
                         '2',
                         '3',
                         '4',
                         '5',
                         '6',
                         '7',
                         '8',
                         '9',
                         '10',
                         '11',
                         '12',
                         '13',
                         '14',
                         '15',
                         '16',
                         '17',
                         '18',
                         '19',
                         '20',
                         '21',
                         '22',
                         '23',
                         '24',
                         '25',
                         '26',
                         '27',
                         '28',
                         '29',
                         '30',
                         '31',
                         '32',
                         '33',
                         '34',
                         '35',
                         '36',
                         '37',
                         '38',
                         '39',
                         '40'],
             'file_index': 0,
             'file_name': '18.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0,
                          'values': {'1': '1.1',
                                     '10': '3.8',
                                     '11': '3.5',
                                     '12': '2.8',
                                     '13': '4.1',
                                     '14': '2.9',
                                     '15': '5.4',
                                     '16': '5.8',
                                     '17': '2.6',
                                     '18': '4.9',
                                     '19': '3.4',
                                     '2': '2.1',
                                     '20': '3.6',
                                     '21': '5.6',
                                     '22': '0.9',
                                     '23': '1',
                                     '24': '0.6',
                                     '25': '5.1',
                                     '26': '4.8',
                                     '27': '5.3',
                                     '28': '5.9',
                                     '29': '4.9',
                                     '3': '3',
                                     '30': '3',
                                     '31': '4.8',
                                     '32': '1.2',
                                     '33': '4',
                                     '34': '1.3',
                                     '35': '5.7',
                                     '36': '3.4',
                                     '37': '2.8',
                                     '38': '2',
                                     '39': '4.8',
                                     '4': '1',
                                     '40': '3',
                                     '5': '0.7',
                                     '6': '5',
                                     '7': '3',
                                     '8': '3.5',
                                     '9': '4.4',
                                     'Process': 'BI'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import sys

def solve_problem(CSVQA_FRAMES):
    table_id = 'file_0_view_0'
    frame = CSVQA_FRAMES[table_id]
    task_cols = [str(i) for i in range(1, 41)]
    T = task_cols
    P = ['1', '2', '3']
    bi_row = None
    for (idx, row) in frame.iterrows():
        if isinstance(row['Process'], str) and row['Process'].casefold() == 'bi':
            bi_row = row
            break
    if bi_row is None:
        raise ValueError("No row with Process == 'BI' found in 18.csv (file_0_view_0)")
    b_t = {}
    for t in T:
        val = bi_row[t]
        try:
            b_t[t] = float(val)
        except Exception:
            raise ValueError(f'Invalid or missing BI value for task {t}: {val}')
    f_p = {'1': 1.33, '2': 2.0, '3': 2.66}
    m = gp.Model('TaskAssignmentMinMakespan')
    x_vars = m.addVars(T, P, vtype=gp.GRB.BINARY, name='')
    C_max = m.addVar(lb=0.0, name='C_max')
    m.addConstrs((gp.quicksum((x_vars[t, p] for p in P)) == 1 for t in T), name='')
    for p in P:
        m.addConstr(gp.quicksum((b_t[t] / f_p[p] * x_vars[t, p] for t in T)) <= C_max, name=f'ProcLoad_{p}')
    m.setObjective(C_max, gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for t in T:
            for p in P:
                print(f'x[{t},{p}] {x_vars[t, p].VarName} {x_vars[t, p].X}')
        print(f'C_max {C_max.VarName} {C_max.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem(CSVQA_FRAMES)