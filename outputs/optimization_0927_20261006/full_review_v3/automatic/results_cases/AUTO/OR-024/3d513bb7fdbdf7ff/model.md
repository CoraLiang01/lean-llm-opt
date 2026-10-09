#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products with identifiers starting with ‘S700_’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Demand and Inventory Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]

**Data Mapping:**
- All data is sourced from table_id file_0_view_0 (SampleSalesData.csv).
- Index set $I$ is defined by all rows where ‘Product Name’ has prefix ‘S700_’.
- Parameters $A_i$, $d_i$, $I_i$ are mapped from columns ‘Revenue’, ‘Demand’, and ‘Initial Inventory’, respectively, for each $i \in I$.