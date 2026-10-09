[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which suppliers to activate and how much each supplier should ship to each store, in order to meet all store demands for Adidas products at minimum total cost. The total cost includes both fixed activation costs for suppliers and per-unit transportation costs from suppliers to stores.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Facility Location (Uncapacitated Facility Location) problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (indexed by \( i \)), from `fixed_cost.csv` and `transportation_costs.csv` rows (e.g., S1, S2, ..., S6).
    - Stores/Customers (indexed by \( j \)), from `demand.csv` and `transportation_costs.csv` columns (e.g., C1, C2, ..., C6).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Quantity of Adidas product shipped from supplier \( i \) to store \( j \). Type: GRB.CONTINUOUS (nonnegative real numbers).
    -   `y[i]` = 1 if supplier \( i \) is activated (operational), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed activation costs for each supplier: from `fixed_cost.csv`, column `fixed_costs`, keyed by supplier (`Unnamed: 0`).
    -   Per-unit transportation costs: from `transportation_costs.csv`, columns `C1`–`C6` for each supplier row.
    -   Store demands: from `demand.csv`, column `demand`, keyed by customer (`customer`).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    -   The fixed activation cost for each supplier that is activated: \(\sum_{i} \text{fixed\_costs}[i] \cdot y[i]\)
    -   The total transportation cost for all shipments: \(\sum_{i,j} \text{transportation\_costs}[i,j] \cdot x[i,j]\)
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each store \( j \), the total quantity received from all suppliers must equal its demand: \(\sum_{i} x[i,j] = \text{demand}[j]\)
    -   Supplier Activation Linking: For each supplier \( i \) and store \( j \), shipments from supplier \( i \) to store \( j \) are only allowed if supplier \( i \) is activated: \(x[i,j] \leq M \cdot y[i]\), where \( M \) is a sufficiently large constant (e.g., sum of all demands).
    -   Nonnegativity: \(x[i,j] \geq 0\) for all \( i, j \).
    -   Binary Activation: \(y[i] \in \{0,1\}\) for all \( i \).
[Abstract Model Plan END]