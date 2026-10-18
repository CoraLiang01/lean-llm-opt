##### Abstract Model  

**Index Set**  
\(I = \{i \mid \text{product } i \text{ in the retail dataset}\}\)

**Parameters**  
\(w_i \;(\text{Revenue of product } i)\)  
\(d_i \;(\text{Demand of product } i)\)  
\(c_i \;(\text{Initial Inventory of product } i)\)

**Decision Variables**  
\(x_i \in \mathbb{Z},\; x_i \ge 1 \quad \forall i \in I\)

**Objective Function**  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

**Constraints**  

1. **Inventory Constraints**  
\[
x_i \;\le\; c_i \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \;\le\; d_i \quad \forall i \in I
\]

##### Data Mapping  

| Source | Table ID | Column Name | Parameter |
|--------|----------|-------------|-----------|
| Retail sales data | `RetailSalesDataset.csv` | `Revenue` | \(w_i\) |
| Retail sales data | `RetailSalesDataset.csv` | `Demand` | \(d_i\) |
| Retail sales data | `RetailSalesDataset.csv` | `Initial Inventory` | \(c_i\) |
| Retail sales data | `RetailSalesDataset.csv` | `Product Name` | Index \(i\) |

*Only the structure and column identifiers are shown; no record values or counts are included.*