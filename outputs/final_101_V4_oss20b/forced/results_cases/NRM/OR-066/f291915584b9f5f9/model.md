**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & Z = \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij} \;=\; d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
& x_{ij} \;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(activation)}\\[4pt]
& x_{ij} \;\ge\; 0 \qquad &&\forall i\in I,\; j\in J\\[4pt]
& y_i \;\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of suppliers (e.g., \(\{S1,S2\}\))
- \(J\): set of customers (e.g., \(\{C1,C2\}\))

**Parameters**

- \(f_i\): fixed activation cost of supplier \(i\)
- \(d_j\): demand of customer \(j\)
- \(c_{ij}\): unit transportation cost from supplier \(i\) to customer \(j\)

**Variables**

- \(y_i\): binary variable, 1 if supplier \(i\) is activated
- \(x_{ij}\): non‑negative continuous variable, quantity supplied from \(i\) to \(j\)

---

### Data Mapping Rules

| Parameter | Source Table | Source Column | Indexing |
|-----------|--------------|---------------|----------|
| \(f_i\)   | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` (supplier id) |
| \(d_j\)   | `file_0_view_0` | `demand` | `customer` (customer id) |
| \(c_{ij}\)| `file_2_view_0` | `C1`, `C2`, … | Row `Unnamed: 0` (supplier id), column `C1`, `C2`, … (customer id) |

These mappings bind each symbolic parameter to the exact column in the corresponding CSV view, preserving the original identifiers and data structure. No numerical values are listed; the model remains fully symbolic.