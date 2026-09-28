#### Abstract Mathematical Model

**Index Sets:**
- $P$: Set of products (indexed by $p$), as identified in 36-1.csv and 36-3.csv.
- $L$: Set of production lines (indexed by $l$), as identified in 36-2.csv.

**Parameters:**
- $D_p$: Maximum demand for product $p$ (units of 100 kg).  
  [36-1.csv, column "Maximum Demand (100 kg units)"]
- $S_p$: Selling price of product $p$ ($/100 kg).  
  [36-1.csv, column "Selling Price ($/100 kg)"]
- $C_p$: Production cost of product $p$ ($/100 kg).  
  [36-1.csv, column "Production Cost ($/100 kg)"]
- $Q_p$: Maximum daily production quota for product $p$ (units of 100 kg per day).  
  [36-1.csv, column "Production Quota (max per day)"]
- $F_l$: Fixed activation cost for production line $l$ ($).  
  [36-2.csv, column "Activation Cost ($)"]
- $B_p$: Minimum batch size for product $p$ (units of 100 kg).  
  [36-3.csv, column "Minimum Batch Size (100 kg units)"]
- $T$: Number of production days in the planning horizon (given as $T = 22$).

**Decision Variables:**
- $x_p \in \mathbb{Z}_+$: Quantity of product $p$ to produce (units of 100 kg).
- $y_l \in \{0,1\}$: 1 if production line $l$ is activated, 0 otherwise.

**Objective:**
\[
\max \left\{ \sum_{p \in P} (S_p - C_p) x_p - \sum_{l \in L} F_l y_l \right\}
\]

**Constraints:**

1. **Demand Constraint:**  
   $\forall p \in P: \quad x_p \leq D_p$

2. **Production Quota Constraint:**  
   $\forall p \in P: \quad x_p \leq Q_p \cdot T$

3. **Minimum Batch Size Constraint:**  
   $\forall p \in P: \quad x_p = 0 \quad \text{or} \quad x_p \geq B_p$

4. **Production-Activation Linking Constraint:**  
   For each product $p$, let $l(p)$ denote the production line associated with $p$ (if each product has its own line):  
   $\forall p \in P: \quad x_p \leq D_p \cdot y_{l(p)}$  
   (If the mapping is not one-to-one, adjust accordingly.)

5. **Variable Domains:**  
   $\forall p \in P: \quad x_p \in \mathbb{Z}_+$  
   $\forall l \in L: \quad y_l \in \{0,1\}$

---

#### Data Mapping

- **36-1.csv**  
  - Table ID: file_0_view_0  
  - Columns:  
    - "Maximum Demand (100 kg units)" $\rightarrow D_p$  
    - "Selling Price ($/100 kg)" $\rightarrow S_p$  
    - "Production Cost ($/100 kg)" $\rightarrow C_p$  
    - "Production Quota (max per day)" $\rightarrow Q_p$  
    - Product identifiers: "A1", ..., "A80"

- **36-2.csv**  
  - Table ID: file_1_view_0  
  - Column:  
    - "Activation Cost ($)" $\rightarrow F_l$  
    - Production line identifiers: "A1", ..., "A80"

- **36-3.csv**  
  - Table ID: file_2_view_0  
  - Column:  
    - "Minimum Batch Size (100 kg units)" $\rightarrow B_p$  
    - Product identifiers: "A1", ..., "A80"

---

**Notes:**
- The mapping between products and production lines is assumed to be one-to-one based on identifiers ("A1" to "A80").
- All quantities are in integer multiples of 100 kg.
- The planning horizon is 22 days.
- The model maximizes total profit, accounting for revenue, production costs, and fixed activation costs, and enforces minimum batch size and integer constraints.