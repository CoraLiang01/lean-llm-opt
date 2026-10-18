[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each vehicle type to store in a set of warehouses so that the total value of stored vehicles is maximised while respecting the capacity limits of each warehouse. The decision variables must be integers.

2.  **Identify Model Type:** This is a Mixed‑Integer Programming (MIP) problem – integer decision variables with a linear objective and linear capacity constraints.

3.  **Define Index Sets:**  
   - **Products** \(i \in I\) – the 10 vehicle types listed in *products.csv*.  
   - **Warehouses** \(w \in W\) – the 10 warehouses listed in *capacity.csv*.

4.  **Define Decision Variables:**  
   - `x[i,w]` = number of vehicles of product \(i\) stored in warehouse \(w\).  
     Type: `GRB.INTEGER`, \(x[i,w] \ge 0\).  
   - (Optional) `x[i]` = total number of vehicles of product \(i\) stored across all warehouses, defined as \(x[i] = \sum_{w \in W} x[i,w]\).  
     Type: `GRB.INTEGER`, \(x[i] \ge 0\).

5.  **Identify Parameters (from Schema):**  
   - **Value[i]** – benefit coefficient (value per vehicle) from the *Value* column of *products.csv*.  
   - **Weight[i]** – space occupied per vehicle from the *Weight* column of *products.csv*.  
   - **Capacity[w]** – maximum space available in warehouse \(w\) from the *Capacity* column of *capacity.csv*.

6.  **Formulate Objective:**  
   Maximise total value of stored vehicles:  
   \[
   \max \sum_{i \in I}\sum_{w \in W} \text{Value}[i] \times x[i,w]
   \]

7.  **Formulate Constraints:**  
   - **Warehouse Capacity Constraints** – for each warehouse \(w\):  
     \[
     \sum_{i \in I} \text{Weight}[i] \times x[i,w] \;\le\; \text{Capacity}[w]
     \]
   - **Non‑negativity & Integrality** – \(x[i,w] \in \mathbb{Z}_{\ge 0}\) for all \(i,w\).  
   - (Optional) **Total‑per‑product definition** – if the auxiliary variable \(x[i]\) is used:  
     \[
     x[i] = \sum_{w \in W} x[i,w] \quad \forall i \in I
     \]

[Abstract Model Plan END]