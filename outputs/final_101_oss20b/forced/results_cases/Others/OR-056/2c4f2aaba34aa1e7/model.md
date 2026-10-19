[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign boats of various types to multiple display areas so that the total monetary value of all boats displayed is maximised, while ensuring that the combined weight of boats in each display area does not exceed that area’s capacity.  
2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MILP)** problem (integer decision variables, linear objective and constraints).  
3.  **Define Index Sets:**  
    - **I** = {1, 2, …, 14} – the set of display areas (from `capacity.csv`).  
    - **J** = {1, 2, …, 20} – the set of boat types (from `products.csv`).  
4.  **Define Decision Variables:**  
    - `x[i,j]` = number of boats of type *j* placed in display area *i*.  
      Type: **GRB.INTEGER** (non‑negative).  
5.  **Identify Parameters (from Schema):**  
    - `Capacity[i]` – capacity of display area *i* (column `Capacity` in `capacity.csv`).  
    - `Value[j]` – monetary value of one boat of type *j* (column `Value` in `products.csv`).  
    - `Weight[j]` – weight of one boat of type *j* (column `Weight` in `products.csv`).  
6.  **Formulate Objective:**  
    - **Maximise**  
      \[
      \sum_{i \in I}\;\sum_{j \in J}\; \text{Value}[j] \times x[i,j]
      \]  
      (total value of all boats displayed).  
7.  **Formulate Constraints:**  
    - **Capacity Constraint (for each display area):**  
      \[
      \sum_{j \in J}\; \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i] \quad \forall i \in I
      \]  
    - **Non‑negativity (implicit in integer type):**  
      \[
      x[i,j] \;\ge\; 0 \quad \forall i \in I,\; j \in J
      \]  
    - **(Optional) Upper bound on total boats per area** – not specified, so omitted.  
[Abstract Model Plan END]