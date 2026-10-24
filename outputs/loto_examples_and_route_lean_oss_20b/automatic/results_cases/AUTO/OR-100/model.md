```markdown
**Sets**  
```
W  : workshops  
C  : components
```

**Parameters**  
```
processing_time[w,c]   : time per unit of component c in workshop w  
unit_price[c]          : unit price of component c  
total_hours[w]         : available working hours in workshop w
```

**Decision variables**  
```
x[c]  : number of units of component c produced  (integer, ≥ 0)
```

**Objective**  
```
maximize  Σ_{c∈C} unit_price[c] · x[c]
```

**Constraints**  
```
for all w∈W:
    Σ_{c∈C} processing_time[w,c] · x[c]  ≤  total_hours[w]

x[c] ∈ ℤ_+   for all c∈C
```

---

### Data Mapping (CSVQA_DATA)

| Parameter / Set | Source Table | Key Columns | Notes |
|-----------------|--------------|-------------|-------|
| `W`             | `file_0_view_0` | `Unnamed: 0` (row id) | Workshops: Casting, Milling, Finishing, Assembly, QA & Packaging |
| `C`             | `file_0_view_0` | `C1`–`C111` (column ids) | 111 component identifiers |
| `processing_time[w,c]` | `file_0_view_0` | row id `Unnamed: 0`, column id `C1`–`C111` | Matrix of processing times |
| `unit_price[c]` | `file_1_view_0` | `Unnamed: 0` (component id), `unit_price` | Unit price per component |
| `total_hours[w]` | `file_2_view_0` | `workshop` (row id), `total_hours` | Available hours per workshop |
```