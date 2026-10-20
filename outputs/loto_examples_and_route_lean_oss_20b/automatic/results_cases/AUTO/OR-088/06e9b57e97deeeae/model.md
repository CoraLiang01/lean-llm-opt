[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which of the 15 candidate plants to build and how much each opened plant should ship to each of the 15 customers so that all customer demands are met, plant capacities are not exceeded, and the total cost (fixed opening costs plus per‑unit transportation costs) is minimized.

2.  **Identify Model Type:** This is a Mixed Integer Programming (MIP) problem with fixed‑charge (opening) and transportation costs – a Capacitated Fixed‑Charge Transportation / Facility Location problem.

3.  **Define Index Sets:**  
   - **Plants**: \(I = \{F1, F2, \dots, F15\}\)  
   - **Customers**: \(J = \{C1, C2, \dots, C15\}\)

4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity shipped from plant *i* to customer *j*.  
      Type: `GRB.CONTINUOUS`, \(x[i,j] \ge 0\).  
    - `y[i]` = 1 if plant *i* is opened, 0 otherwise.  
      Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
    - **Fixed opening cost**: `fixed_cost[i]` from column `fixed_cost` in *cost.csv*.  
    - **Plant capacity**: `capacity[i]` from column `capacity` in *cost.csv*.  
    - **Per‑unit transport cost**: `transport_cost[i,j]` from columns `C1`–`C15` in *cost.csv*.  
    - **Customer demand**: `demand[j]` from column `demand` in *demand.csv*.

6.  **Formulate Objective:**  
   Minimize  
   \[
   \sum_{i \in I} \text{fixed_cost}[i] \cdot y[i]
   \;+\;
   \sum_{i \in I}\sum_{j \in J} \text{transport_cost}[i,j] \cdot x[i,j].
   \]

7.  **Formulate Constraints:**  
    - **Demand satisfaction** (for each customer \(j\)):  
      \[
      \sum_{i \in I} x[i,j] \;=\; \text{demand}[j].
      \]
    - **Capacity linking** (for each plant \(i\)):  
      \[
      \sum_{j \in J} x[i,j] \;\le\; \text{capacity}[i] \cdot y[i].
      \]
      (If \(y[i]=0\) the RHS is 0, forcing all shipments from that plant to be 0.)  
    - **Non‑negativity**: \(x[i,j] \ge 0\) for all \(i,j\).  
    - **Binary**: \(y[i] \in \{0,1\}\) for all \(i\).

[Abstract Model Plan END]