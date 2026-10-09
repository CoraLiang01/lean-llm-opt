##### Mathematical Optimization Model

**Index Set:**
- $I$: set of all products (from "Product Name" in table_id: file_0_view_0)

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from "Revenue", table_id: file_0_view_0)
- $d_i$: deterministic demand for product $i$ (from "Demand", table_id: file_0_view_0)
- $I_i$: initial inventory for product $i$ (from "Initial Inventory", table_id: file_0_view_0)

**Decision Variables:**
- $x_i$: number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq I_i, && \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, && \forall i \in I
\end{align*}
\]

##### Data Mapping

- $I$: All "Product Name" entries in table_id: file_0_view_0, column "Product Name" (MobileSalesDataset.csv)
- $A_i$: "Revenue" column in table_id: file_0_view_0 (MobileSalesDataset.csv)
- $d_i$: "Demand" column in table_id: file_0_view_0 (MobileSalesDataset.csv)
- $I_i$: "Initial Inventory" column in table_id: file_0_view_0 (MobileSalesDataset.csv)