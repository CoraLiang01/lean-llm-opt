import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    workers = ['Carpenter', 'Electrician', 'Painter', 'Worker_004', 'Worker_005', 'Worker_006', 'Worker_007', 'Worker_008', 'Worker_009', 'Worker_010', 'Worker_011', 'Worker_012', 'Worker_013', 'Worker_014', 'Worker_015', 'Worker_016', 'Worker_017', 'Worker_018', 'Worker_019', 'Worker_020', 'Worker_021', 'Worker_022', 'Worker_023', 'Worker_024', 'Worker_025', 'Worker_026', 'Worker_027', 'Worker_028', 'Worker_029', 'Worker_030', 'Worker_031', 'Worker_032', 'Worker_033', 'Worker_034', 'Worker_035', 'Worker_036', 'Worker_037', 'Worker_038', 'Worker_039', 'Worker_040', 'Worker_041', 'Worker_042', 'Worker_043', 'Worker_044', 'Worker_045', 'Worker_046', 'Worker_047', 'Worker_048', 'Worker_049', 'Worker_050', 'Worker_051', 'Worker_052', 'Worker_053', 'Worker_054', 'Worker_055', 'Worker_056', 'Worker_057', 'Worker_058', 'Worker_059', 'Worker_060', 'Worker_061', 'Worker_062', 'Worker_063', 'Worker_064', 'Worker_065', 'Worker_066', 'Worker_067', 'Worker_068', 'Worker_069', 'Worker_070', 'Worker_071', 'Worker_072', 'Worker_073', 'Worker_074', 'Worker_075', 'Worker_076', 'Worker_077', 'Worker_078', 'Worker_079', 'Worker_080', 'Worker_081', 'Worker_082', 'Worker_083', 'Worker_084', 'Worker_085', 'Worker_086', 'Worker_087', 'Worker_088', 'Worker_089', 'Worker_090', 'Worker_091', 'Worker_092', 'Worker_093', 'Worker_094', 'Worker_095', 'Worker_096', 'Worker_097', 'Worker_098', 'Worker_099', 'Worker_100', 'Worker_101', 'Worker_102', 'Worker_103', 'Worker_104', 'Worker_105', 'Worker_106', 'Worker_107', 'Worker_108', 'Worker_109', 'Worker_110', 'Worker_111', 'Worker_112', 'Worker_113', 'Worker_114', 'Worker_115', 'Worker_116', 'Worker_117', 'Worker_118', 'Worker_119', 'Worker_120', 'Worker_121', 'Worker_122', 'Worker_123', 'Worker_124', 'Worker_125', 'Worker_126', 'Worker_127', 'Worker_128', 'Worker_129', 'Worker_130', 'Worker_131', 'Worker_132', 'Worker_133', 'Worker_134', 'Worker_135', 'Worker_136', 'Worker_137', 'Worker_138', 'Worker_139', 'Worker_140', 'Worker_141', 'Worker_142', 'Worker_143', 'Worker_144', 'Worker_145', 'Worker_146', 'Worker_147', 'Worker_148', 'Worker_149', 'Worker_150']
    N = len(workers)
    work_days = []
    for i in range(N):
        row = {'Owner': workers[i]}
        for j in range(N):
            row[workers[j]] = 10 if i == j else 0
        work_days.append(row)
    if len(work_days) != N:
        raise ValueError('Number of rows in work_days does not match number of workers')
    for row in work_days:
        for w in workers:
            if w not in row:
                raise ValueError(f"Missing entry for worker {w} in row {row['Owner']}")
    D = {}
    for row in work_days:
        owner = row['Owner']
        D[owner] = {}
        for w in workers:
            D[owner][w] = row[w]
    m = gp.Model('mutual_wage')
    m.Params.MIPGap = 0.0001
    w = m.addVars(workers, lb=0.0, vtype=GRB.CONTINUOUS, name='')
    m.addConstr(w[workers[0]] == 60.0, name='wage_norm')
    for i in workers:
        lhs = gp.LinExpr()
        for j in workers:
            if j != i:
                lhs += D[i][j] * w[j]
        rhs = gp.LinExpr()
        for j in workers:
            rhs += D[j][i]
        rhs = rhs * w[i]
        m.addConstr(lhs == rhs, name='')
    for j in workers:
        total_days = sum((D[i][j] for i in workers))
        m.addConstr(total_days == 10, name='')
    m.setObjective(gp.quicksum((w[j] for j in workers)), GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')