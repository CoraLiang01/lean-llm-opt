**Mathematical Model**

```
Sets
    M   : managers
    P   : projects

Parameters
    c[m,p]   : cost for manager m to manage project p   (m∈M, p∈P)

Decision Variables
    x[m,p]   ∈ {0,1}   (m∈M, p∈P)   – 1 if manager m is assigned to project p

Objective
    minimize   Z = Σ_{m∈M} Σ_{p∈P} c[m,p] · x[m,p]

Constraints
    (1)  Manager‑to‑project assignment
         Σ_{p∈P} x[m,p] = 1          ∀ m∈M

    (2)  Project‑to‑manager assignment
         Σ_{m∈M} x[m,p] = 1          ∀ p∈P
```

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| `c[m,p]` | Table `file_0_view_0` | For each row where `Unnamed: 0 = m` and each column `Pj = p`, set `c[m,p]` to the numeric value in that cell. |

*Indices*  
- `M` is the set of distinct values in column `Unnamed: 0` of `file_0_view_0`.  
- `P` is the set of column names in `file_0_view_0` excluding `Unnamed: 0`.  

All other model components are defined symbolically; no numerical values are enumerated.