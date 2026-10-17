## Mathematical Model

**Index set**

\[
I=\{\,i\mid \text{product }i\text{ is a 4U product in the filtered view}\,\}
\]

**Parameters** (all taken from table *file_0_view_0*)

\[
\begin{aligned}
d_i &\;=\;\text{Demand of product }i \quad(\text{units})\\
I_i &\;=\;\text{Initial Inventory of product }i \quad(\text{units})\\
r_i &\;=\;\text{Revenue per unit of product }i \quad(\$)
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
x_i &\;\le\; d_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; I_i &&\forall i\in I \quad\text{(cannot sell more than inventory)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Parameter | Source Table | Column Name | Mapping |
|-----------|--------------|-------------|---------|
| \(d_i\) | `file_0_view_0` | `Demand` | \(d_i = \text{Demand}_{\text{row }i}\) |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` | \(I_i = \text{Initial Inventory}_{\text{row }i}\) |
| \(r_i\) | `file_0_view_0` | `Revenue` | \(r_i = \text{Revenue}_{\text{row }i}\) |
| Index set \(I\) | `file_0_view_0` | *row identifier* | \(I = \{\text{row }1,\dots,\text{row }18\}\) |

*All rows in `file_0_view_0` satisfy the filter condition “Product Name contains ‘4U’”. No other rows are included in the model.*