**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}[i]\;x[i] \\[4pt]
\text{subject to}\quad
& 0 \le x[i] \le \text{Demand}[i] && \forall i\in I \\[2pt]
& 0 \le x[i] \le \text{InitialInventory}[i] && \forall i\in I \\[4pt]
& x[i]\in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of Aalop products (one element per row in the source table whose *Product Name* starts with “Aalop”).

**Parameters**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(\text{Revenue}[i]\) | Revenue per unit of product \(i\) | `Revenue` column of table **file_0_view_0** |
| \(\text{Demand}[i]\) | Known demand for product \(i\) | `Demand` column of table **file_0_view_0** |
| \(\text{InitialInventory}[i]\) | Initial inventory of product \(i\) | `Initial Inventory` column of table **file_0_view_0** |

**Decision Variables**

- \(x[i]\): number of units of product \(i\) to sell (integer, non‑negative).

---

**Data Mapping Rules**

1. **Index Set \(I\)**  
   \[
   I = \{\, \text{Product Name} \mid \text{Product Name} \text{ starts with “Aalop”} \,\}
   \]
   (derived from rows of table **file_0_view_0** that satisfy the filter in the query).

2. **Parameter \(\text{Revenue}[i]\)**  
   \[
   \text{Revenue}[i] = \text{value in column “Revenue” of the row where } \text{Product Name}=i
   \]

3. **Parameter \(\text{Demand}[i]\)**  
   \[
   \text{Demand}[i] = \text{value in column “Demand” of the row where } \text{Product Name}=i
   \]

4. **Parameter \(\text{InitialInventory}[i]\)**  
   \[
   \text{InitialInventory}[i] = \text{value in column “Initial Inventory” of the row where } \text{Product Name}=i
   \]

These mappings bind every parameter to the exact columns of the source table **file_0_view_0** and ensure the model uses the data supplied by the observation.