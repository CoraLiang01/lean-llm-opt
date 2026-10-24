**Mathematical Model**

Let  

\[
I=\{\,i \mid \text{Product Name}_i \text{ contains the string “4U”}\,\}
\]

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product }i,\\
d_i &\;=\; \text{Demand of product }i,\\
I_i &\;=\; \text{Initial Inventory of product }i .
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
x_i &\;\le\; I_i &&\forall i\in I \quad\text{(inventory limit)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Index | Source Table | Source Column |
|-------------------|--------------|---------------|
| \(I\) (index set) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |

All parameter values are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.