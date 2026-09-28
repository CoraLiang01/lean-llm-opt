import gurobipy as gp
from gurobipy import GRB
participants = ['Carpenter', 'Electrician', 'Painter', 'Worker_004', 'Worker_005', 'Worker_006', 'Worker_007', 'Worker_008', 'Worker_009', 'Worker_010', 'Worker_011', 'Worker_012', 'Worker_013', 'Worker_014', 'Worker_015', 'Worker_016', 'Worker_017', 'Worker_018', 'Worker_019', 'Worker_020', 'Worker_021', 'Worker_022', 'Worker_023', 'Worker_024', 'Worker_025', 'Worker_026', 'Worker_027', 'Worker_028', 'Worker_029', 'Worker_030', 'Worker_031', 'Worker_032', 'Worker_033', 'Worker_034', 'Worker_035', 'Worker_036', 'Worker_037', 'Worker_038', 'Worker_039', 'Worker_040', 'Worker_041', 'Worker_042', 'Worker_043', 'Worker_044', 'Worker_045', 'Worker_046', 'Worker_047', 'Worker_048', 'Worker_049', 'Worker_050', 'Worker_051', 'Worker_052', 'Worker_053', 'Worker_054', 'Worker_055', 'Worker_056', 'Worker_057', 'Worker_058', 'Worker_059', 'Worker_060', 'Worker_061', 'Worker_062', 'Worker_063', 'Worker_064', 'Worker_065', 'Worker_066', 'Worker_067', 'Worker_068', 'Worker_069', 'Worker_070', 'Worker_071', 'Worker_072', 'Worker_073', 'Worker_074', 'Worker_075', 'Worker_076', 'Worker_077', 'Worker_078', 'Worker_079', 'Worker_080', 'Worker_081', 'Worker_082', 'Worker_083', 'Worker_084', 'Worker_085', 'Worker_086', 'Worker_087', 'Worker_088', 'Worker_089', 'Worker_090', 'Worker_091', 'Worker_092', 'Worker_093', 'Worker_094', 'Worker_095', 'Worker_096', 'Worker_097', 'Worker_098', 'Worker_099', 'Worker_100', 'Worker_101', 'Worker_102', 'Worker_103', 'Worker_104', 'Worker_105', 'Worker_106', 'Worker_107', 'Worker_108', 'Worker_109', 'Worker_110', 'Worker_111', 'Worker_112', 'Worker_113', 'Worker_114', 'Worker_115', 'Worker_116', 'Worker_117', 'Worker_118', 'Worker_119', 'Worker_120', 'Worker_121', 'Worker_122', 'Worker_123', 'Worker_124', 'Worker_125', 'Worker_126', 'Worker_127', 'Worker_128', 'Worker_129', 'Worker_130', 'Worker_131', 'Worker_132', 'Worker_133', 'Worker_134', 'Worker_135', 'Worker_136', 'Worker_137', 'Worker_138', 'Worker_139', 'Worker_140', 'Worker_141', 'Worker_142', 'Worker_143', 'Worker_144', 'Worker_145', 'Worker_146', 'Worker_147', 'Worker_148', 'Worker_149', 'Worker_150']
N = len(participants)
d = {}
for i, owner in enumerate(participants):
    d[owner] = {}
    for j, worker in enumerate(participants):
        if i == 0 and j == 0:
            d[owner][worker] = 1
        elif i == 0 and j == 1:
            d[owner][worker] = 0
        elif i == 0 and j == 2:
            d[owner][worker] = 0
        elif i == 0 and j == N - 1:
            d[owner][worker] = 1
        elif i == 1 and j == 0:
            d[owner][worker] = 1
        elif i == 1 and j == 1:
            d[owner][worker] = 1
        elif i == 1 and j == 2:
            d[owner][worker] = 1
        elif i == 2 and j == 1:
            d[owner][worker] = 1
        elif i == 2 and j == 2:
            d[owner][worker] = 1
        else:
            d[owner][worker] = 0
if set(d.keys()) != set(participants):
    raise ValueError('Mismatch in owner keys of d and participants')
for owner in participants:
    if set(d[owner].keys()) != set(participants):
        raise ValueError(f'Mismatch in worker keys of d[{owner}] and participants')
m = gp.Model('mutual_wage')
m.Params.MIPGap = 0.0001
w = m.addVars(participants, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='')
m.addConstr(w[participants[0]] == 60.0, name='wage_norm')
for k in participants:
    lhs = gp.quicksum((d[i][k] * w[i] for i in participants))
    rhs = gp.quicksum((d[k][j] * w[j] for j in participants))
    m.addConstr(lhs == rhs, name='balance_' + k)
m.setObjective(gp.quicksum((w[j] * w[j] for j in participants)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')