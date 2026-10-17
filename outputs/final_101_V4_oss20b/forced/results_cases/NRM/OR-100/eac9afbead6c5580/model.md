**Mathematical Model**

```
Sets
C : set of components
W : set of workshops

Parameters
t[w,c]   : processing time per unit of component c in workshop w
p[c]     : unit price of component c
H[w]     : total available working hours for workshop w

Decision variables
x[c]  ∈ ℤ_+   (production quantity of component c)

Objective
maximise   ∑_{c∈C} p[c] · x[c]

Constraints
∑_{c∈C} t[w,c] · x[c]  ≤  H[w]          ∀ w∈W          (work‑shop capacity)
x[c]  ≥ 0,  integer   ∀ c∈C
```

---

**Data Mapping**

| Parameter | Source Table (table_id) | Source Columns | Notes |
|-----------|--------------------------|----------------|-------|
| `t[w,c]`  | `file_0_view_0` (processing_time_unit.csv) | Row identifier: `"Unnamed: 0"` (workshop names); Column identifier: `"C1"`–`"C111"` (component IDs) | Each cell gives processing time for component *c* in workshop *w*. |
| `p[c]`    | `file_1_view_0` (unit_price.csv) | Row identifier: `"Unnamed: 0"` (component IDs); Column: `"unit_price"` | Unit price of component *c*. |
| `H[w]`    | `file_2_view_0` (total_working_hours.csv) | Row identifier: `"workshop"` (workshop names); Column: `"total_hours"` | Total available hours for workshop *w*. |

*Sets*  
- `C` is the set of all component identifiers appearing in `unit_price.csv` (`C1`–`C111`).  
- `W` is the set of workshop names appearing in `processing_time_unit.csv` (`Casting`, `Milling`, `Finishing`, `Assembly`, `QA & Packaging`).  

All numerical values are taken directly from the corresponding source tables; no additional data are introduced.