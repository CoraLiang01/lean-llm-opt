#### Abstract Mathematical Model

**Index Sets:**
- $P$: Set of products (from A1 to A80).

**Parameters:**
- $D_p$: Maximum demand for product $p$ (units of 100 kg).  
  [table_id: file_0_view_0, column: A* where Product = "Maximum Demand (100 kg units)"]
- $S_p$: Selling price of product $p$ ($/100 kg).  
  [table_id: file_0_view_0, column: A* where Product = "Selling Price ($/100 kg)"]
- $C_p$: Production cost of product $p$ ($/100 kg).  
  [table_id: file_0_view_0, column: A* where Product = "Production Cost ($/100 kg)"]
- $Q_p$: Maximum daily production quota for product $p$ (units of 100 kg per day).  
  [table_id: file_0_view_0, column: A* where Product = "Production Quota (max per day)"]
- $F_p$: Fixed activation cost for product $p$ ($).  
  [table_id: file_1_view_0, column: A* where Product = "Activation Cost ($)"]
- $B_p$: Minimum batch size for product $p$ (units of 100 kg).  
  [table_id: file_2_view_0, column: A* where Product = "Minimum Batch Size (100 kg units)"]
- $T$: Number of production days in the planning horizon (given as 22).

**Decision Variables:**
- $x_p \in \mathbb{Z}_+$: Number of 100 kg units of product $p$ to produce in the month.
- $y_p \in \{0,1\}$: 1 if production line for product $p$ is activated, 0 otherwise.

**Objective:**
\[
\max \sum_{p \in P} \left[ S_p \cdot x_p - C_p \cdot x_p - F_p \cdot y_p \right]
\]

**Constraints:**

1. **Demand Constraint:**  
   $\forall p \in P: \quad x_p \leq D_p$

2. **Production Capacity Constraint:**  
   $\forall p \in P: \quad x_p \leq Q_p \cdot T$

3. **Minimum Batch Size Constraint:**  
   $\forall p \in P: \quad x_p \geq B_p \cdot y_p$

4. **Activation Linking Constraint:**  
   $\forall p \in P: \quad x_p \leq (Q_p \cdot T) \cdot y_p$

5. **Integrality:**  
   $\forall p \in P: \quad x_p \in \mathbb{Z}_+, \quad y_p \in \{0,1\}$

---

#### Data Mapping

- **file_0_view_0 (36-1.csv):**
  - Product = "Maximum Demand (100 kg units)" → $D_p$
  - Product = "Selling Price ($/100 kg)" → $S_p$
  - Product = "Production Cost ($/100 kg)" → $C_p$
  - Product = "Production Quota (max per day)" → $Q_p$
- **file_1_view_0 (36-2.csv):**
  - Product = "Activation Cost ($)" → $F_p$
- **file_2_view_0 (36-3.csv):**
  - Product = "Minimum Batch Size (100 kg units)" → $B_p$

All products $p$ correspond to columns A1, A2, ..., A80 in each table. The planning horizon $T$ is given as 22 days. All variables are integer as required.