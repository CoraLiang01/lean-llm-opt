**Mathematical model**

```
SETS
    W   :  {A,B,C,D,E,F,G,H,I,J}          ! workers
    T   :  {1,2,3,4,5,6,7,8,9,10}        ! tasks to be assigned

PARAMETERS
    time[w,t]   >= 0   ! hours worker w needs for task t

VARIABLES
    x[w,t]      binary   ! 1 if worker w is assigned to task t

MINIMIZE
    total_hours = sum_{w in W} sum_{t in T} time[w,t] * x[w,t]

CONSTRAINTS
    ! each task gets exactly one worker
    task_assign[t] : sum_{w in W} x[w,t] = 1          for all t in T

    ! each worker does at most one task
    worker_use[w]  : sum_{t in T} x[w,t] <= 1          for all w in W
```

**Data‑mapping rules**

| Parameter | Source table | Mapping expression |
|-----------|--------------|--------------------|
| `time[w,t]` | `file_0_view_0` (15.csv) | `time[w,t] = value in row where "Task Time Required" = t and column = w` |

*Only rows with `"Task Time Required"` values 1–10 are used; rows 11–12 are ignored.*