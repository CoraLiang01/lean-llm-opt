**Mathematical Model**

Let  

- \(O\) be the set of all generation lot options (the values of the column *option* in *energy.csv*).  
- For each \(o\in O\)  
  - \(g_o\)  – generation per lot (column *gen_per_lot*).  
  - \(c_o\)  – cost per lot (column *cost_per_lot*).  

Decision variable  

- \(x_o \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(o\) to purchase.

---

\[
\begin{aligned}
\min \quad & \sum_{o\in O} c_o\,x_o \\[4pt]
\text{s.t.}\quad & \sum_{o\in O} g_o\,x_o \;\ge\; 200 \\[4pt]
& x_o \in \mathbb{Z}_{\ge 0}\quad \forall\,o\in O
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(g_o\) | generation per lot | `file_0_view_0` | `gen_per_lot` |
| \(c_o\) | cost per lot | `file_0_view_0` | `cost_per_lot` |

*Note:* The column `unit_cost_est` is present in the source file but is not used in the model. The set \(O\) is derived from all distinct values of the `option` column in the complete `energy.csv` dataset.