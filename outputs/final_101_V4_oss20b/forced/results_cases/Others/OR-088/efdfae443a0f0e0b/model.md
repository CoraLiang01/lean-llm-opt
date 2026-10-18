[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which of 15 candidate plants to build and how to ship products from the opened plants to 15 customers so that all customer demands are met, plant capacities are respected, and the total cost (fixed opening costs plus per‑unit shipping costs) is minimized.

2.  **Identify Model Type:** This is a **Capacitated Facility Location** problem, a Mixed‑Integer Linear Program (MIP) with fixed‑charge (opening) costs.

3.  **Define Index Sets:**  
   - **Plants** \(P = \{F1, F2, \dots, F15\}\)  
   - **Customers** \(C = \{C1, C2, \dots, C15\}\)

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity shipped from plant *i* to customer *j*.  
     Type: `GRB.CONTINUOUS`, \(x_{i,j} \ge 0\).  
   - `y[i]` = 1 if plant *i* is opened, 0 otherwise.  
     Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
   - **Fixed opening cost**: `fixed_cost[i]` from `cost.csv` column *fixed_cost*.  
   - **Plant capacity**: `capacity[i]` from `cost.csv` column *capacity*.  
   - **Per‑unit transport cost**: `transport_cost[i,j]` from `cost.csv` columns *C1*–*C15*.  
   - **Customer demand**: `demand[j]` from `demand.csv` column *demand*.

6.  **Formulate Objective:**  
   Minimize  
   \[
   \sum_{i \in P} \text{fixed_cost}[i] \cdot y[i]
   \;+\;
   \sum_{i \in P}\sum_{j \in C} \text{transport_cost}[i,j] \cdot x[i,j].
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction** (for each customer \(j\)):  
     \[
     \sum_{i \in P} x[i,j] \;=\; \text{demand}[j].
     \]
   - **Capacity linking** (for each plant \(i\)):  
     \[
     \sum_{j \in C} x[i,j] \;\le\; \text{capacity}[i] \cdot y[i].
     \]
   - **Non‑negativity**:  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i \in P,\, j \in C.
     \]
   - **Binary opening decisions**:  
     \[
     y[i] \in \{0,1\} \quad \forall i \in P.
     \]

[Abstract Model Plan END]