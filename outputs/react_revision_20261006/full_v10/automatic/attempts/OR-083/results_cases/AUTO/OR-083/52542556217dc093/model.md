Mathematical Model

Sets:
W = {1, 2, ..., 12}   (workers, indexed by w; corresponds to "Task Time Required" rows 1–12 in file_0_view_0)
T = {A, B, C, D, E, F, G, H, I, J}   (tasks, indexed by t; corresponds to columns A–J in file_0_view_0)

Parameters:
c_{w,t} = time required for worker w to complete task t
  (Data Mapping: file_0_view_0, row w, column t; w ∈ W, t ∈ T)

Decision Variables:
x_{w,t} ∈ {0,1}  (w ∈ W, t ∈ T)
  x_{w,t} = 1 if worker w is assigned to task t, 0 otherwise

Objective:
minimize  ∑_{w∈W} ∑_{t∈T} c_{w,t} x_{w,t}

Subject to:
1. Each task is assigned to exactly one worker:
  ∑_{w∈W} x_{w,t} = 1  ∀ t ∈ T

2. Each worker is assigned to at most one task:
  ∑_{t∈T} x_{w,t} ≤ 1  ∀ w ∈ W

3. Exactly 10 workers are assigned (i.e., 2 workers are not assigned any task):
  ∑_{w∈W} ∑_{t∈T} x_{w,t} = 10

4. x_{w,t} ∈ {0,1}  ∀ w ∈ W, t ∈ T

Data Mapping:
Set W: file_0_view_0, "Task Time Required" rows 1–12 (worker indices 1–12)
Set T: file_0_view_0, columns A–J (task labels)
Parameter c_{w,t}: file_0_view_0, value at row w, column t (w ∈ W, t ∈ T)