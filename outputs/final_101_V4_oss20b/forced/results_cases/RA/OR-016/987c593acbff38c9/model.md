**Mathematical Model**

Let  

* \(I\) be the set of all products in *RetailSalesDataset.csv* (identified by the column **Product Name**).  

Parameters (all taken directly from the table `file_0_view_0`):  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{value in column “Revenue” for product } i,\\
\text{Demand}_i  &\;=\; \text{value in column “Demand” for product } i,\\
\text{Inventory}_i &\;=\; \text{value in column “Initial Inventory” for product } i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

(Units of product \(i\) to sell.)

---

**Objective**

\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

---

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; \text{Demand}_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; \text{Inventory}_i \qquad \forall i\in I
   \]

(These two inequalities together enforce \(x_i \le \min(\text{Demand}_i,\text{Inventory}_i)\).)

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| Revenue\_i | `file_0_view_0` | `Revenue` |
| Demand\_i  | `file_0_view_0` | `Demand` |
| Inventory\_i | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the distinct values in the column **Product Name** of the same table. No other data are required.