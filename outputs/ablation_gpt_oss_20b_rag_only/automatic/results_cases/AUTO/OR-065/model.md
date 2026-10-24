**Mathematical Model (Facility‑Location Formulation)**  

```
Sets
  I  : set of warehouses  (S1,S2,S3)
  J  : set of customers   (C1,C2,C3)

Parameters
  f[i]          : fixed opening cost of warehouse i
  c[i,j]        : unit transportation cost from warehouse i to customer j
  d[j]          : demand of customer j

Decision Variables
  y[i] ∈ {0,1}          : 1 if warehouse i is opened, 0 otherwise
  x[i,j] ≥ 0            : quantity shipped from i to j

Objective
  minimize  Σ_{i∈I} f[i]·y[i]  +  Σ_{i∈I} Σ_{j∈J} c[i,j]·x[i,j]

Constraints
  (1) Demand satisfaction
      Σ_{i∈I} x[i,j]  ≥  d[j]          ∀ j∈J

  (2) Shipments only from open warehouses
      x[i,j]  ≤  d[j] · y[i]            ∀ i∈I, j∈J

  (3) Variable domains
      y[i] ∈ {0,1}                      ∀ i∈I
      x[i,j] ≥ 0                        ∀ i∈I, j∈J
```

---

**Data‑Mapping Rules**

| Parameter | Source Table | Index Column | Value Column | Index Set |
|-----------|--------------|--------------|--------------|-----------|
| `f[i]`    | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | `I` |
| `c[i,j]`  | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3` | `I × J` |
| `d[j]`    | `file_0_view_0` | `customer`   | `demand`      | `J` |

*All indices (`i` and `j`) are taken from the complete set of rows returned in the observation; no additional filtering is applied.*