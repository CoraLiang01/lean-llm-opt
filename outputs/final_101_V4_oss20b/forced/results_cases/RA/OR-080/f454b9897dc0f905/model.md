**Mathematical Model**

```
Sets
  I   : set of trucks  (truck_id from file_0_view_0)
  T   : {1,2,3,4}

Parameters
  Q_i   : maximum capacity of truck i          (file_0_view_0, column Q)
  S_i   : startup cost of truck i              (file_0_view_0, column S)
  C_i   : unit transport cost of truck i       (file_0_view_0, column C)
  D_t   : demand in period t
          D_1 = d1, D_2 = d2, D_3 = d3, D_4 = d4   (file_0_view_0)
  Δ_max : 300   (change‑limit constant)
  β      : 0.9   (spare‑capacity buffer)

Variables
  x_{i,t}  ≥ 0          (continuous, weight transported by truck i in period t)
  y_{i,t}  ∈ {0,1}      (binary, truck i active in period t)
  z_{i,t}  ∈ {0,1}      (binary, truck i started in period t)
  w_{i,t}  ∈ {0,1}      (binary, truck i shut down in period t)   (t = 2..4)

Objective
  minimize  Σ_{i∈I} Σ_{t∈T} ( S_i · z_{i,t} + C_i · x_{i,t} )

Constraints
  1. Startup limits
     z_{i,t} ≤ y_{i,t}                                 ∀i∈I, t∈T
     z_{i,1} ≥ y_{i,1}                                 ∀i∈I
     z_{i,t} ≥ y_{i,t} – y_{i,t-1}                     ∀i∈I, t=2..4

  2. Minimum consecutive active periods
     y_{i,t+1} ≥ y_{i,t} + z_{i,t} – 1                 ∀i∈I, t=1..3

  3. No start in the last period
     z_{i,4} = 0                                       ∀i∈I

  4. Shutdown rules
     w_{i,t} ≥ y_{i,t-1} – y_{i,t}                     ∀i∈I, t=2..4
     w_{i,t} ≤ y_{i,t-1}                               ∀i∈I, t=2..4
     w_{i,t} ≤ 1 – y_{i,t}                             ∀i∈I, t=2..4
     y_{i,t+1} ≤ 1 – w_{i,t}                           ∀i∈I, t=2..3
     y_{i,t+2} ≤ 1 – w_{i,t}                           ∀i∈I, t=2

  5. Capacity limits
     x_{i,t} ≤ Q_i · y_{i,t}                           ∀i∈I, t∈T

  6. Change‑limit between consecutive periods
     x_{i,t} – x_{i,t-1} ≤ Δ_max                       ∀i∈I, t=1..4
     x_{i,t-1} – x_{i,t} ≤ Δ_max                       ∀i∈I, t=1..4
     (with x_{i,0} = 0 for all i)

  7. Demand satisfaction
     Σ_{i∈I} x_{i,t} ≥ D_t                              ∀t∈T

  8. Spare‑capacity buffer
     Σ_{i∈I} x_{i,t} ≤ β · Σ_{i∈I} Q_i · y_{i,t}       ∀t∈T
```

---

### Data Mapping

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| `I` | `file_0_view_0` | `truck_id` | Set of truck identifiers |
| `T` | — | `{1,2,3,4}` | Explicit period set |
| `Q_i` | `file_0_view_0` | `Q` | Max capacity of truck i |
| `S_i` | `file_0_view_0` | `S` | Startup cost of truck i |
| `C_i` | `file_0_view_0` | `C` | Unit transport cost of truck i |
| `D_1` | `file_0_view_0` | `d1` | Demand period 1 |
| `D_2` | `file_0_view_0` | `d2` | Demand period 2 |
| `D_3` | `file_0_view_0` | `d3` | Demand period 3 |
| `D_4` | `file_0_view_0` | `d4` | Demand period 4 |
| `Δ_max` | — | constant `300` | Change‑limit |
| `β` | — | constant `0.9` | Spare‑capacity buffer |
| `x_{i,0}` | — | constant `0` | Initial load |

All other constants (e.g., binary domain, non‑negativity) are implicit in the variable definitions.