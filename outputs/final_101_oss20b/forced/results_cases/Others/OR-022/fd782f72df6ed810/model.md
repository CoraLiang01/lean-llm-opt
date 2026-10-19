[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each 27‑inch product should be sold during a promotional event so that total revenue is maximized, while respecting the available inventory and the known demand for each product.

2.  **Identify Model Type:** This is a **Linear/Integer Programming** problem (a simple MIP if integer sales are required, otherwise an LP).

3.  **Define Index Sets:**  
   - **Products** \(i \in \mathcal{P}\) – the set of all products whose *Product Name* contains the substring “27in” (e.g., “27in 4K Gaming Monitor”, “27in FHD Monitor”).

4.  **Define Decision Variables:**  
   - `x[i]` = number of units of product *i* that will be sold during the event.  
     Type: **GRB.INTEGER** (use GRB.CONTINUOUS if fractional sales are acceptable).  
     Bounds: \(x[i] \ge 0\).

5.  **Identify Parameters (from Schema):**  
   - **Revenue**: `Revenue[i]` – the revenue per unit for product *i* (from the *Revenue* column).  
   - **Initial Inventory**: `Inv[i]` – the number of units of product *i* available at the start of the event (from the *Initial Inventory* column).  
   - **Demand**: `Dem[i]` – the maximum number of units of product *i* that customers are willing to buy (from the *Demand* column).

6.  **Formulate Objective:**  
   \[
   \text{Maximize } \sum_{i \in \mathcal{P}} \text{Revenue}[i] \times x[i]
   \]

7.  **Formulate Constraints:**  
   - **Inventory Constraint** (cannot sell more than what is in stock):  
     \[
     \forall i \in \mathcal{P}:\; x[i] \;\le\; \text{Inv}[i]
     \]
   - **Demand Constraint** (cannot sell more than the known demand):  
     \[
     \forall i \in \mathcal{P}:\; x[i] \;\le\; \text{Dem}[i]
     \]
   - **Non‑negativity** (already implied by variable type):  
     \[
     \forall i \in \mathcal{P}:\; x[i] \;\ge\; 0
     \]
[Abstract Model Plan END]