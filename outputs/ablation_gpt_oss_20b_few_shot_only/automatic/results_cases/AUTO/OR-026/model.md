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

**Index Sets**

- \(I\) : set of all products whose category is **Fashion** (i.e., product names beginning with “Fashion”).

**Parameters**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(\text{Revenue}_i\) | Revenue per unit of product \(i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`, column **Revenue** |
| \(\text{Demand}_i\) | Demand quantity for product \(i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`, column **Demand** |
| \(\text{InitialInventory}_i\) | Initial inventory for product \(i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`, column **Initial Inventory** |

**Data Mapping Rules**

- **Set \(I\)**:  
  \[
  I = \{\, i \mid \text{ProductName}_i \text{ starts with the string } "Fashion" \,\}
  \]
  where \(\text{ProductName}_i\) is the value in the **Product Name** column of the CSV.

- **Parameter \(\text{Revenue}_i\)**:  
  Map each row \(i\) to the numeric value in the **Revenue** column.

- **Parameter \(\text{Demand}_i\)**:  
  Map each row \(i\) to the numeric value in the **Demand** column.

- **Parameter \(\text{InitialInventory}_i\)**:  
  Map each row \(i\) to the numeric value in the **Initial Inventory** column.

All mappings use the row index of the CSV as the identifier for \(i\). No additional data or constraints are introduced beyond those specified.