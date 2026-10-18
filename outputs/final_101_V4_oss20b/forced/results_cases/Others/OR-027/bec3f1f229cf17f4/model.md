[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to maximize total revenue by deciding how many units of each *Organ* product to sell, subject to the available initial inventory and the known deterministic demand for those products.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem (continuous decision variables; no binary or integer decisions are required unless the user explicitly specifies integer units).

3.  **Define Index Sets:**  
   - **Products (P)**: the set of all sub‑categories whose *Sub Category* field contains the string “Organ” (e.g., “Organic Fruits”, “Organic Staples”, “Organic Vegetables”).  
   (If the data uses the exact word “Organ” instead of “Organic”, the filter should match that exact string.)

4.  **Define Decision Variables:**  
   - `x[i]` = number of units of product *i* to sell (for all `i ∈ P`).  
     Type: `GRB.CONTINUOUS` (use `GRB.INTEGER` if units must be whole numbers).

5.  **Identify Parameters (from Schema):**  
   - `Revenue[i]` – revenue per unit of product *i* (from the **Revenue** column).  
   - `Demand[i]` – maximum units that can be sold (from the **Demand** column).  
   - `Inventory[i]` – available units in stock (from the **Initial Inventory** column).

6.  **Formulate Objective:**  
   - **Maximize**  
     \[
     \sum_{i \in P} \text{Revenue}[i] \times x[i]
     \]

7.  **Formulate Constraints:**  
   - **Demand limit** (cannot sell more than the known demand):  
     \[
     x[i] \;\le\; \text{Demand}[i] \quad \forall i \in P
     \]
   - **Inventory limit** (cannot sell more than what is in stock):  
     \[
     x[i] \;\le\; \text{Inventory}[i] \quad \forall i \in P
     \]
   - **Non‑negativity**:  
     \[
     x[i] \;\ge\; 0 \quad \forall i \in P
     \]
[Abstract Model Plan END]