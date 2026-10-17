**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{Inventory}_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0 && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| Revenue\_i | file_0_view_0 | Revenue |
| Demand\_i | file_0_view_0 | Demand |
| Inventory\_i | file_0_view_0 | Initial Inventory |

**Index Set**

- \(I\): set of product names from *RestaurantSalesreport.csv* whose *Product Name* starts with the prefix “Aalop”.