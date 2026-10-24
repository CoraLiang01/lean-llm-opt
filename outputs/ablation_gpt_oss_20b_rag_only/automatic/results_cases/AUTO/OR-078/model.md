**Mathematical Model**

Let  

* \(O\) – set of all generation‑lot options listed in *energy.csv*  
* \(g_o\) – generation per lot (kWh) for option \(o\)  
* \(c_o\) – cost per lot (USD) for option \(o\)  

Decision variables  

* \(x_o \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(o\) to purchase  

Objective (minimise total procurement cost)

\[
\min \;\; \sum_{o\in O} c_o\,x_o
\]

Demand‑coverage constraint (total generation must meet the required 200 kWh)

\[
\sum_{o\in O} g_o\,x_o \;\ge\; 200
\]

No other constraints are imposed.

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(g_o\) | Generation per lot | `file_0_view_0` | `gen_per_lot` |
| \(c_o\) | Cost per lot | `file_0_view_0` | `cost_per_lot` |

*Index set*  
\(O = \{\,\text{option}\;|\;\text{option} \text{ appears in table } \text{file_0_view_0}\,\}\)

All rows of the table are used; no filtering is applied.  
The decision set \(x_o\) is defined over the same index set \(O\).  
All parameters are taken directly from the corresponding columns of the table.