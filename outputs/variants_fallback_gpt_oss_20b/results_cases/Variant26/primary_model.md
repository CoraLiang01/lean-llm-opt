**Mathematical Model**

```
Sets
    W   : set of workers with on_leave = 0
    P   : set of projects
    O   : set of admissible worker–project pairs (w,p) ∈ W×P
          for which a cost entry exists and skill_rank[w] ≥ required_rank[p]

Parameters
    skill_rank[w]   ∈ {1,2,3,4}          (Junior=1, Intermediate=2, Senior=3, Expert=4)
    required_rank[p]∈ {1,2,3,4}
    cost[w,p]       ∈ ℕ,  (w,p)∈O

Decision Variables
    x[w,p] ∈ {0,1}   for (w,p)∈O

Objective
    minimize   Σ_{(w,p)∈O}  cost[w,p] · x[w,p]

Constraints
    1.  Assignment of projects
        ∀ p∈P :  Σ_{w : (w,p)∈O}  x[w,p]  = 1

    2.  Worker capacity
        ∀ w∈W :  Σ_{p : (w,p)∈O}  x[w,p]  ≤ 1
```

---

**Data‑Mapping Rules**

| Parameter / Set | Source Table | Source Columns | Mapping Rule |
|-----------------|--------------|----------------|--------------|
| `W` | `file_0_view_0` | `worker_id`, `on_leave` | `W = { w | on_leave[w] = 0 }` |
| `P` | `file_1_view_0` | `project_id` | `P = { p | project_id[p] exists }` |
| `skill_rank[w]` | `file_0_view_0` | `skill` | `skill_rank[w] = 1` if skill=`Junior`; `2` if `Intermediate`; `3` if `Senior`; `4` if `Expert` |
| `required_rank[p]` | `file_1_view_0` | `required_skill` | `required_rank[p] = 1` if required_skill=`Junior`; `2` if `Intermediate`; `3` if `Senior`; `4` if `Expert` |
| `cost[w,p]` | `file_2_view_0` | `worker_id`, `project_id`, `cost_cents` | `cost[w,p] = cost_cents` for each record |
| `O` | Derived | – | `O = { (w,p) | (w,p)∈cost  ∧  skill_rank[w] ≥ required_rank[p] }` |

*All numeric values are taken directly from the CSV files; no enumeration of the table rows is shown.*