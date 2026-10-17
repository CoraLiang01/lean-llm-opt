## Mathematical Model

**Index set**

\[
I=\{i\mid i \text{ is a pizza type in the dataset}\}
\]

**Parameters**

\[
\begin{aligned}
r_i &\;=\;\text{Revenue per unit of pizza }i\\
d_i &\;=\;\text{Demand for pizza }i\\
inv_i &\;=\;\text{Initial inventory of pizza }i
\end{aligned}
\]

**Decision variables**

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(demand limit)}\\
x_i &\le inv_i &&\forall i\in I &&\text{(inventory limit)}
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(inv_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |

The index set \(I\) is the set of all rows returned in `file_0_view_0`. Each row corresponds to a distinct pizza type identified by the `Product Name` column, which is implicitly used as the index identifier. No additional filtering or aggregation is applied.