[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many units of each vehicle type to store in a set of warehouses so that the total value of the stored cars is maximised, while ensuring that the weight of cars in each warehouse does not exceed that warehouse’s capacity. All decision variables must be integers.

2. **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem – linear objective, linear constraints, integer decision variables.

3. **Define Index Sets:**  
   - **Products** \(I\): the 10 vehicle types listed in *products.csv* (Sedans, SUVs, Electric Vehicles, …).  
   - **Warehouses** \(W\): the 10 warehouses listed in *capacity.csv* (Warehouse 1 … Warehouse 10).

4. **Define Decision Variables:**  
   - `x[i,w]` = number of vehicles of product *i* stored in warehouse *w*.  
     *Type:* `GRB.INTEGER` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - **Value[i]** – the benefit coefficient (value) of product *i*, from the `Value` column of *products.csv*.  
   - **Weight[i]** – the weight (or space) consumed by one unit of product *i*, from the `Weight` column of *products.csv*.  
   - **Capacity[w]** – the maximum weight capacity of warehouse *w*, from the `Capacity` column of *capacity.csv*.

6. **Formulate Objective:**  
   Maximise total value of stored cars:  
   \[
   \max \sum_{i\in I}\sum_{w\in W} \text{Value}[i] \times x[i,w]
   \]

7. **Formulate Constraints:**  
   - **Warehouse capacity limits** (for each warehouse \(w\)):  
     \[
     \sum_{i\in I} \text{Weight}[i] \times x[i,w] \;\le\; \text{Capacity}[w]
     \]
   - **Non‑negativity and integrality** (for all \(i,w\)):  
     \[
     x[i,w] \;\ge\; 0 \quad \text{and integer}
     \]

[Abstract Model Plan END]