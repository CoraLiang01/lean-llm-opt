**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad 
& x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{InitialInventory}_i && \forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | *rows* |
| \(\text{Revenue}_i\) | Parameter | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | Parameter | `file_0_view_0` | `Demand` |
| \(\text{InitialInventory}_i\) | Parameter | `file_0_view_0` | `Initial Inventory` |

*All indices \(i\) correspond to the three rows returned for “Organ” products.*