**Mathematical Model**

Let  

- \(I\) – set of pizza types (indexed by the *Product Name* values in the source table).  

Parameters (all indexed by \(i\in I\))  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of pizza }i \quad (\text{column “Revenue”})\\
d_i &\;=\;\text{Demand of pizza }i \quad (\text{column “Demand”})\\
s_i &\;=\;\text{Initial inventory of pizza }i \quad (\text{column “Initial Inventory”})
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
x_i &\;\le\; d_i &&\forall i\in I \quad &&\text{(demand limit)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad &&\text{(inventory limit)}
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Source Column |
|--------|--------------|---------------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| \(I\)   | `file_0_view_0` | `Product Name` (each distinct value forms an index element) |

All parameters are taken directly from the specified columns of the single source table `file_0_view_0`. No additional data or constraints are introduced beyond those required by the query.