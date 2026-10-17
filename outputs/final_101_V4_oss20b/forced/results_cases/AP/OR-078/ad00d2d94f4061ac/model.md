**Mathematical Model**

Let  

* \(O\) – set of all generation lot options (the *option* column of *energy.csv*).  
* For each \(o\in O\):  
  * \(g_o\) – generation per lot (from *gen_per_lot*).  
  * \(c_o\) – cost per lot (from *cost_per_lot*).  

Decision variable  

* \(x_o \in \mathbb{Z}_{+}\) – number of lots of option \(o\) to purchase.

---

\[
\begin{aligned}
\min \quad & \sum_{o\in O} c_o\,x_o \\[4pt]
\text{s.t.}\quad & \sum_{o\in O} g_o\,x_o \;\ge\; 200 \\[4pt]
& x_o \;\in\; \mathbb{Z}_{+}\qquad \forall\,o\in O
\end{aligned}
\]

---

**Data Mapping**

| Source | Mapping Rule |
|--------|--------------|
| `file_0_view_0` (energy.csv) | For each record \(r\):<br>• \(g_{r.option} \gets r.gen_per_lot\)<br>• \(c_{r.option} \gets r.cost_per_lot\) |
| `tech` column | Not used in the model (kept for reference). |

All parameters are taken directly from the complete set of 131 rows in *energy.csv*; no values are enumerated in the model.