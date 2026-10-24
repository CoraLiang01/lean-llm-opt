**Mathematical Model**

\[
\begin{aligned}
\text{Sets} \quad & I \;=\;\{\text{Product\_Reference}\mid \text{Product\_Reference starts with "ELE-S"}\}\\[4pt]

\text{Parameters} \quad & r_i \;=\;\text{Revenue of product }i,\\
& d_i \;=\;\text{Demand of product }i,\\
& \text{inv}_i \;=\;\text{Initial Inventory of product }i, \qquad i\in I\\[4pt]

\text{Decision Variables} \quad & x_i \;\in\;\mathbb{Z}_{\ge 0}, \qquad i\in I\\[4pt]

\text{Objective} \quad & \max \;\sum_{i\in I} r_i\,x_i\\[4pt]

\text{Constraints} \quad & 0 \;\le\; x_i \;\le\; d_i, \qquad i\in I\\
& 0 \;\le\; x_i \;\le\; \text{inv}_i, \qquad i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product_Reference` (filtered by prefix `"ELE-S"`) |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the specified columns of the table `file_0_view_0`. No additional data or assumptions are introduced.