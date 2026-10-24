**Mathematical Model**

Let  

- \(I\) be the set of distinct FDK57 car‑model records (index \(i\)).  
- \(r_i\) – revenue per unit of record \(i\).  
- \(d_i\) – deterministic demand for record \(i\).  
- \(v_i\) – initial inventory for record \(i\).  
- \(x_i\) – quantity of record \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit |
| \(d_i\) | `file_0_view_0` | `Demand` | Deterministic demand |
| \(v_i\) | `file_0_view_0` | `Initial Inventory` | Available inventory |

| Index Set | Source Table | Identifier | Mapping |
|-----------|--------------|------------|---------|
| \(I\) | `file_0_view_0` | `data_row_number` | Each distinct `data_row_number` in the returned rows defines a unique index \(i\). |

The mapping preserves the raw identifiers from the CSV; no numerical values are enumerated in the model.