**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{dem}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{inv}_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source File | Column Name | Description |
|--------|-------------|-------------|-------------|
| \(I\) | Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv | `Product_Name` | All rows where `Product_Name` starts with `"Books_"` |
| \(\text{rev}_i\) | Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv | `Revenue` | Revenue per unit for product \(i\) |
| \(\text{dem}_i\) | Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv | `Demand` | Demand quantity for product \(i\) |
| \(\text{inv}_i\) | Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv | `Initial Inventory` | Initial inventory for product \(i\) |
| \(x_i\) | Decision variable | – | Units of product \(i\) to fulfill (integer, \(\ge 0\)) |

*All parameters are mapped directly from the specified columns of the CSV file; no additional data are introduced.*