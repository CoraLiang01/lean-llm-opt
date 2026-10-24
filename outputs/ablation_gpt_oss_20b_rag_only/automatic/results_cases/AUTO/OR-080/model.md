**Mathematical model**

```
SETS
    I   : set of trucks  (truck_id values from the table)
    T   : {1,2,3,4}      (time periods)
    T0  : {0}            (initial period)

PARAMETERS
    Q[i]   : maximum capacity of truck i          (kg)
    S[i]   : startup cost of truck i              ($)
    C[i]   : unit transport cost of truck i       ($/kg)
    d[t]   : customer demand in period t          (kg)

VARIABLES
    y[i,t]  ∈ {0,1}   truck i is active in period t
    s[i,t]  ∈ {0,1}   truck i starts up in period t
    z[i,t]  ∈ {0,1}   truck i shuts down in period t  (t=1..3)
    w[i,t]  ≥ 0       weight transported by truck i in period t

CONSTRAINTS
    1.  y[i,0] = 0                                 ∀ i∈I

    2.  Startup definition
        s[i,t] ≥ y[i,t] - y[i,t-1]                 ∀ i∈I, t∈T
        s[i,t] ≤ y[i,t]                           ∀ i∈I, t∈T
        s[i,t] ≤ 1 - y[i,t-1]                     ∀ i∈I, t∈T
        s[i,4] = 0                                 ∀ i∈I

    3.  Minimum two‑consecutive‑active periods
        y[i,t+1] ≥ y[i,t] - y[i,t-1]               ∀ i∈I, t=1..3

    4.  Shutdown definition
        z[i,t] ≥ y[i,t-1] - y[i,t]                 ∀ i∈I, t=1..3
        z[i,t] ≤ y[i,t-1]                         ∀ i∈I, t=1..3
        z[i,t] ≤ 1 - y[i,t]                       ∀ i∈I, t=1..3

    5.  Inactivity after shutdown
        y[i,t+1] ≤ 1 - z[i,t]                     ∀ i∈I, t=1..3
        y[i,t+2] ≤ 1 - z[i,t]                     ∀ i∈I, t=1..2

    6.  Weight capacity
        w[i,t] ≤ Q[i] * y[i,t]                    ∀ i∈I, t∈T
        w[i,t] ≥ 0                                 ∀ i∈I, t∈T

    7.  Weight change limit (300 kg)
        w[i,t] - w[i,t-1] ≤ 300                   ∀ i∈I, t∈T
        w[i,t-1] - w[i,t] ≤ 300                   ∀ i∈I, t∈T
        w[i,0] = 0                                 ∀ i∈I

    8.  Demand satisfaction
        Σ_i w[i,t] ≥ d[t]                         ∀ t∈T

    9.  Spare‑capacity buffer (90 %)
        Σ_i w[i,t] ≤ 0.9 * Σ_i Q[i] * y[i,t]      ∀ t∈T

OBJECTIVE
    minimize  Σ_i Σ_t ( S[i] * s[i,t] + C[i] * w[i,t] )
```

All indices in the constraints belong to the declared sets.  
The only fixed value is `w[i,0] = 0` for all trucks.

---

**Data‑mapping rules**

| Parameter | Source table | Column(s) | Mapping rule |
|-----------|--------------|-----------|--------------|
| `Q[i]`   | `file_0_view_0` | `Q` | `Q[i] = table[truck_id=i].Q` |
| `S[i]`   | `file_0_view_0` | `S` | `S[i] = table[truck_id=i].S` |
| `C[i]`   | `file_0_view_0` | `C` | `C[i] = table[truck_id=i].C` |
| `d[1]`   | `file_0_view_0` | `d1` | `d[1] = table[truck_id=1].d1` |
| `d[2]`   | `file_0_view_0` | `d2` | `d[2] = table[truck_id=1].d2` |
| `d[3]`   | `file_0_view_0` | `d3` | `d[3] = table[truck_id=1].d3` |
| `d[4]`   | `file_0_view_0` | `d4` | `d[4] = table[truck_id=1].d4` |

*Indices*  
`I` is the set of all `truck_id` values present in `file_0_view_0`.  
`T` is the fixed set `{1,2,3,4}`.  
`T0` is `{0}` for the initial period.

These mappings supply all parameter values used in the model; no other data are required.