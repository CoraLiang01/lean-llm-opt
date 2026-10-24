**Mathematical Model (Facility‑Location with Transportation)**  

```
SETS
    I   : {i | i ∈ suppliers}
    J   : {j | j ∈ customers}

PARAMETERS
    f[i]   : fixed cost of opening supplier i
    c[i,j] : transportation cost per vehicle from supplier i to customer j
    d[j]   : demand of customer j
    M      : big‑M constant  (e.g. M = Σ_j d[j])

VARIABLES
    y[i]   : binary, 1 if supplier i is opened, 0 otherwise
    x[i,j] : quantity of vehicles shipped from supplier i to customer j   (x[i,j] ≥ 0)

OBJECTIVE
    minimize  Σ_i f[i]·y[i]  +  Σ_i Σ_j c[i,j]·x[i,j]

CONSTRAINTS
    (1) Demand satisfaction
        ∀ j ∈ J :   Σ_i x[i,j]  ≥  d[j]

    (2) Linking (no shipment if supplier closed)
        ∀ i ∈ I, ∀ j ∈ J :   x[i,j]  ≤  M · y[i]

    (3) Variable domains
        ∀ i ∈ I :   y[i] ∈ {0,1}
        ∀ i ∈ I, ∀ j ∈ J :   x[i,j] ≥ 0
```

---

**Data‑Mapping Rules**

| Parameter | Source Table | Row ID Column | Column ID / Value Column | Notes |
|-----------|--------------|---------------|--------------------------|-------|
| `f[i]` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Supplier identifier is the row ID. |
| `c[i,j]` | `file_2_view_0` | `Unnamed: 0` | `C1` … `C9` | Row ID is supplier (`Unnamed: 0`), column IDs are customer names (`C1`–`C9`). |
| `d[j]` | `file_0_view_0` | `customer` | `demand` | Customer identifier is the row ID. |
| `M` | *derived* | – | – | `M` is defined as the sum of all `d[j]` (or any sufficiently large constant). |

All index sets `I` and `J` are taken from the complete set of suppliers and customers present in the respective tables; no additional filtering is applied unless explicitly requested. The model uses the raw identifiers from the CSV files, preserving the exact mapping between data and decision variables.