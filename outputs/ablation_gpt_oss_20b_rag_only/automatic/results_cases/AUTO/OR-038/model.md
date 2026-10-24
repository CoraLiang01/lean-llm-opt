**Mathematical Model**

```
Sets
    I   – vehicle types

Parameters
    Capacity[i]   – daily inventory limit for type i   (from capacity.csv)
    Value[i]      – benefit coefficient for type i     (from products.csv)

Variables
    x[i]  – number of vehicles of type i ordered per day
            integer,  x[i] ≥ 0

Objective
    maximize  Σ_{i∈I}  Value[i] * x[i]

Constraints
    (1)  Σ_{i∈I}  x[i]  ≤  Σ_{i∈I}  Capacity[i]          (overall capacity)
    (2)  ∀ i∈I :  x[i]  ≤  Capacity[i]                    (per‑type limit)
```

---

**Data Mapping**

| Symbol | Source Table | Column(s) | Join Key |
|--------|--------------|-----------|----------|
| I      | file_0_view_0 | VehicleType | – |
| Capacity[i] | file_0_view_0 | Capacity | VehicleType = i |
| Value[i] | file_1_view_0 | Value | ProductName = i |

*All other columns in the source tables are ignored.*