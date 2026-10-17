**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of produce types (index \(i\) corresponds to the \(i\)-th row of *products.csv*).  
* \(C\) – overall inventory capacity (from *capacity.csv*).  
* \(w_i\) – weight of one unit of product \(i\) (column **Weight**).  
* \(v_i\) – benefit (value) of one unit of product \(i\) (column **Value**).  
* \(x_i\in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | single scalar value |
| \(w_i\) | `file_1_view_0` | `Weight` | indexed by row order (1‑10) |
| \(v_i\) | `file_1_view_0` | `Value` | indexed by row order (1‑10) |
| \(I\) | `file_1_view_0` | – | set of row indices 1…10 |

All indices, parameters, and variables are defined exactly as above; no additional data or aggregation is introduced.