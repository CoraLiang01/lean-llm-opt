**Mathematical Model**

Let  

- \(O\) be the set of all generation options (rows of *energy.csv*).  
- For each \(o\in O\)  
  - \(g_o\)  – generation per lot (units of demand)  
  - \(c_o\)  – cost per lot  

Decision variables  

\[
x_o \in \mathbb{Z}_{\ge 0}\qquad \forall\,o\in O
\]

Objective  

\[
\min \;\sum_{o\in O} c_o\,x_o
\]

Demand‑coverage constraint  

\[
\sum_{o\in O} g_o\,x_o \;\ge\; 200
\]

No other constraints are required.

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(O\) | *energy.csv* | \(O = \{\,\text{option}\,\}\) |
| \(g_o\) | *energy.csv* | \(g_o = \text{gen\_per\_lot}\) for row with option \(o\) |
| \(c_o\) | *energy.csv* | \(c_o = \text{cost\_per\_lot}\) for row with option \(o\) |

*Table ID:* `file_0_view_0`  
*Columns used:* `option`, `gen_per_lot`, `cost_per_lot`