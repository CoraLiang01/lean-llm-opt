#### Abstract Mathematical Optimization Model

**Index Set:**
- $i \in \mathcal{P}$: Set of all baked goods (products) offered by the bakery.

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
- $I_i$: Initial inventory available for product $i$.  
- $d_i$: Deterministic demand for product $i$ over the sales horizon.

**Decision Variables:**
- $x_i$: Quantity of product $i$ to fulfill (sell), $x_i \geq 0$.

**Objective Function:**
\[
\max \sum_{i \in \mathcal{P}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{P}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{P}
   \]
3. **Non-negativity:**  
   \[
   x_i \geq 0 \qquad \forall i \in \mathcal{P}
   \]

**Variable Domain:**  
- $x_i$ can be continuous or integer, as not specified in the question.

---

#### Data Mapping

- $\mathcal{P}$ (Index set): All unique values in column **Product Name** from table_id: `file_0_view_0` (Frenchbakerydailysales.csv).
- $A_i$ (Revenue per unit): Column **Revenue** from table_id: `file_0_view_0`, indexed by **Product Name**.
- $I_i$ (Initial Inventory): Column **Initial Inventory** from table_id: `file_0_view_0`, indexed by **Product Name**.
- $d_i$ (Demand): Column **Demand** from table_id: `file_0_view_0`, indexed by **Product Name**.

All mappings use the CSVQA_DATA bindings as validated above.