**Mathematical Model**

```
SETS
    I   : component types
    W   : workshops

PARAMETERS
    t[w,i]   : processing time per unit of component i in workshop w   (≥0)
    p[i]     : unit price of component i                               (≥0)
    H[w]     : total available working hours in workshop w             (≥0)

VARIABLES
    x[i]     : number of units of component i to produce                (≥0, continuous)

OBJECTIVE
    maximize  Z = Σ_{i∈I}  p[i] * x[i]

CONSTRAINTS
    for all w∈W:
        Σ_{i∈I}  t[w,i] * x[i]  ≤  H[w]
```

**Data Mapping (CSVQA_DATA)**  

| Symbol | Table ID          | Row‑ID column | Column‑ID column | Value column |
|--------|-------------------|---------------|------------------|--------------|
| I      | file_1_view_0     | Unnamed: 0   | –                | –            |
| W      | file_2_view_0     | –             | –                | –            |
| t[w,i] | file_0_view_0     | Unnamed: 0   | Unnamed: 0       | –            |
| p[i]   | file_1_view_0     | –             | –                | unit_price   |
| H[w]   | file_2_view_0     | –             | –                | total_hours  |

*All indices and parameters are taken directly from the corresponding tables; no additional data are introduced.*