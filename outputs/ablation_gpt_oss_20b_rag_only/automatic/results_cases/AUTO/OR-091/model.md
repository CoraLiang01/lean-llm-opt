**Mathematical Model**

```
Sets
  I   : set of Operations‑Research courses  (course identifiers)

Parameters
  c[i]   : credits of course i          (c[i] ≥ 0)
  p[i]   : interest‑points of course i   (p[i] ≥ 0)

Decision Variables
  x[i]   ∈ {0,1}   for all i ∈ I          (1 if course i is selected, 0 otherwise)

Objective
  maximize   Z = Σ_{i∈I}  p[i] · x[i]

Subject to
  CreditLimit : Σ_{i∈I}  c[i] · x[i]  ≤ 20
```

---

**Data Mapping Rules**

| Element | Source | Mapping |
|---------|--------|---------|
| Set `I` | `file_0_view_0` (filtered view) | `I = { course_id | row in file_0_view_0 where discipline = "Operations Research" }` |
| Parameter `c[i]` | `file_0_view_0` | `c[i] = credits` (numeric column) |
| Parameter `p[i]` | `file_0_view_0` | `p[i] = interest_points` (numeric column) |
| Filter condition | `file_0_view_0` | `discipline = "Operations Research"` (exact string match) |

All other data (credits, interest points) are taken directly from the corresponding columns of the filtered table. No additional constraints or variables are introduced.