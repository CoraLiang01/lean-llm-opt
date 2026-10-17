**Mathematical Model**

Let  

- \(O\) be the set of all generation options (rows of *energy.csv*).  
- For each \(o\in O\):  
  - \(g_o\)  – generation per lot (from column **gen_per_lot**)  
  - \(c_o\)  – cost per lot (from column **cost_per_lot**)  
  - \(u_o\)  – unit‑cost estimate (from column **unit_cost_est**) – *not used in the objective*  

Decision variable  

- \(x_o \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(o\) to purchase.

---

\[
\begin{aligned}
\min\;& \sum_{o\in O} c_o\,x_o \\[4pt]
\text{s.t. } & \sum_{o\in O} g_o\,x_o \;\ge\; 200 \\[4pt]
& x_o \in \mathbb{Z}_{\ge 0}\quad \forall\,o\in O
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column(s) | Mapping |
|----------------------|--------------|-----------|---------|
| \(O\) | `file_0_view_0` | `option` | Set of all distinct `option` values |
| \(g_o\) | `file_0_view_0` | `gen_per_lot` | `g_o = gen_per_lot` for each `option` |
| \(c_o\) | `file_0_view_0` | `cost_per_lot` | `c_o = cost_per_lot` for each `option` |
| \(u_o\) | `file_0_view_0` | `unit_cost_est` | `u_o = unit_cost_est` for each `option` |
| \(x_o\) | – | – | Decision variable, integer, non‑negative |

All parameters are taken directly from the complete runtime records of *energy.csv* (`file_0_view_0`). No additional data or constraints are introduced beyond those specified in the query.