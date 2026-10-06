#### Abstract Mathematical Model

**Index Set:**

- $i \in \mathcal{F}$: Set of all products classified under ‘FAUX’ (indexed by Product Name from file_0_view_0).

**Parameters:**

- $A_i$: Revenue per unit of product $i$ (from column Revenue, table_id file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column Initial Inventory, table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column Demand, table_id file_0_view_0).

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in \mathcal{F}$.

**Objective:**

$$
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{F}
   $$
2. **Demand Constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{F}
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{F}
   $$

---

#### Data Mapping

- $\mathcal{F}$: All rows in table_id file_0_view_0 (ZARASales.csv) where Product Name has prefix ‘FAUX’.
- $A_i$: file_0_view_0, column Revenue, indexed by Product Name.
- $I_i$: file_0_view_0, column Initial Inventory, indexed by Product Name.
- $d_i$: file_0_view_0, column Demand, indexed by Product Name.

(CSVQA_DATA bindings: file_0_view_0, columns [Product Name, Revenue, Initial Inventory, Demand], filter Product Name prefix ‘FAUX’)