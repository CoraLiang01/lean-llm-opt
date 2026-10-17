**Symbolic Model**

```
Sets
    W   : set of workers  (columns of the table except the column “Owner”)
    H   : set of homeowners (rows of the table)

Parameters
    d[i,j]   : number of days worker j spent on homeowner i’s home
    owner[i] : worker who is the homeowner of i

Variables
    w[j]     : daily wage of worker j   (real,  w[j] ≥ 0)

Fixed Data
    w[Carpenter] = 60.00

Constraints
    (* Income–expenditure balance for each homeowner *)
    ∀ i ∈ H :  Σ_{j∈W, j ≠ owner[i]}  d[i,j] * w[j]
               = d[i,owner[i]] * w[owner[i]]

    (* Optional consistency check – total days per worker *)
    (* ∀ j ∈ W :  Σ_{i∈H} d[i,j] = 10  *)

(* No objective – the system is a feasibility problem. *)
```

---

**Data‑Mapping Rules**

| Parameter / Variable | Source Table | Source Column | Notes |
|----------------------|--------------|---------------|-------|
| `d[i,j]` | `file_0_view_0` | column `j` (e.g., “Carpenter”, “Electrician”, …, “Worker_150”) | Row `i` corresponds to the `i`‑th record in the CSV. |
| `owner[i]` | `file_0_view_0` | column `Owner` | The value in this column is the name of the worker who is the homeowner of record `i`. |
| `w[Carpenter]` | fixed value | – | Daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. |
| `w[j]` for `j ≠ Carpenter` | – | – | Decision variables to be determined. |

*All other workers (`Electrician`, `Painter`, `Worker_004`, …, `Worker_150`) are included in the set `W` automatically from the column names of the source table.*