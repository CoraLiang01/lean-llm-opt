## Mathematical Model

**Index set**  
\(I\) – set of all pizza types (rows of *PizzaSalesDataset.csv*).

**Parameters**  
\[
\begin{aligned}
r_i &\;=\; \text{Revenue per unit of pizza }i \quad (i\in I)\\
d_i &\;=\; \text{Demand for pizza }i \quad (i\in I)\\
inv_i &\;=\; \text{Initial inventory of pizza }i \quad (i\in I)
\end{aligned}
\]

**Decision variables**  
\[
x_i \;\;=\;\; \text{number of units of pizza }i\text{ to sell} \quad (i\in I)
\]
\(x_i\in\mathbb{Z}_{\ge 0}\).

**Objective**  
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**  
\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I &&\text{(demand limit)}\\
x_i &\;\le\; inv_i &&\forall i\in I &&\text{(inventory limit)}
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(inv_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |

| Variable | Domain |
|----------|--------|
| \(x_i\) | integer, \(x_i\ge 0\) |

All indices \(i\) are taken from the full set of rows returned in *PizzaSalesDataset.csv* (100 rows). No additional filtering or aggregation is applied. The model uses only the parameters and constraints explicitly stated in the query.