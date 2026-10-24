**Mathematical Model**

Let  

- \(I\) – set of all generation lot options (identified by the column *option* in *energy.csv*).  

Parameters (indexed by \(i\in I\))  

- \(g_i\) – generation per lot (column *gen_per_lot*).  
- \(c_i\) – cost per lot (column *cost_per_lot*).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(i\) to purchase.  

Objective  

\[
\min \;\; \sum_{i\in I} c_i\,x_i
\]

Demand‑meeting constraint  

\[
\sum_{i\in I} g_i\,x_i \;\ge\; 200
\]

(Each lot must be purchased in whole units; no upper bound on \(x_i\).)

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(g_i\)   | file_0_view_0 | gen_per_lot |
| \(c_i\)   | file_0_view_0 | cost_per_lot |

The index set \(I\) is derived from all distinct values in the *option* column of *file_0_view_0*.