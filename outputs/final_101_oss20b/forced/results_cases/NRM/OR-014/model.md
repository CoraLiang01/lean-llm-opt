**Mathematical Model**

Let  

- \(I\) – set of pizza types (indexed by the *Product Name* values in the source table).  

Parameters (all indexed by \(i\in I\))  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue per unit of pizza }i \\
d_i &\;=\;\text{Demand for pizza }i \\
s_i &\;=\;\text{Initial inventory of pizza }i
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I \quad\text{(demand limit)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad\text{(inventory limit)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the distinct values in the `Product Name` column of the same table. No other data sources are referenced.