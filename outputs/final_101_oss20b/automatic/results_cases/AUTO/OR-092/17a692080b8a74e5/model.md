[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks that can carry up to 10 units each.  The number of trucks dispatched on each route must be an integer, but each truck may be partially loaded.  All demand must be satisfied exactly, and the total transportation cost (cost per unit of cargo) must be minimized.

2.  **Identify Model Type:** Mixed‑Integer Linear Programming (MIP) – continuous cargo variables coupled with integer truck‑count variables.

3.  **Define Index Sets:**  
   - **S** = {S1, S2, …, S10}  (sources)  
   - **D** = {D1, D2, …, D20}  (destinations)

4.  **Define Decision Variables:**  
   - `x[s,d]` = amount of cargo shipped from source `s` to destination `d`.  
     *Type:* GRB.CONTINUOUS, `x[s,d] ≥ 0`.  
   - `t[s,d]` = number of trucks dispatched from source `s` to destination `d`.  
     *Type:* GRB.INTEGER, `t[s,d] ≥ 0`.

5.  **Identify Parameters (from Schema):**  
   - `supply_units[s]` – available supply at source `s` (from `expanded_sources.csv`).  
   - `demand_units[d]` – required demand at destination `d` (from `expanded_destinations.csv`).  
   - `cost[s,d]` – unit transportation cost from `s` to `d` (from `expanded_cost_matrix.csv`).  
   - `truck_capacity` – constant = 10 units (maximum cargo per truck).

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{d \in D} \text{cost}[s,d] \times x[s,d].
   \]

7.  **Formulate Constraints:**  
   - **Supply limits:**  
     \[
     \forall s \in S:\quad \sum_{d \in D} x[s,d] \;\le\; \text{supply_units}[s].
     \]
   - **Demand satisfaction:**  
     \[
     \forall d \in D:\quad \sum_{s \in S} x[s,d] \;=\; \text{demand_units}[d].
     \]
   - **Truck capacity (partial loading allowed):**  
     \[
     \forall s \in S,\; \forall d \in D:\quad x[s,d] \;\le\; \text{truck_capacity} \times t[s,d].
     \]
   - **Non‑negativity:**  
     \[
     x[s,d] \ge 0,\quad t[s,d] \ge 0.
     \]
   - **Integrality of trucks:**  
     \[
     t[s,d] \in \mathbb{Z}_{\ge 0}.
     \]

[Abstract Model Plan END]