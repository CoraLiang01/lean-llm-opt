[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate boats of various types to multiple display areas so that the total monetary value of the boats shown is maximized, while ensuring that the combined physical size (weight) of boats in each display area does not exceed that area's capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is a pure Integer Linear Programming (ILP) problem (a multi‑knapsack assignment).

3.  **Define Index Sets:**  
    - **I** = {1, 2, …, 14} – the set of display areas (from `capacity.csv`).  
    - **J** = {1, 2, …, 20} – the set of boat types (from `products.csv`).

4.  **Define Decision Variables:**  
    - `x[i,j]` = number of units of boat type *j* placed in display area *i*.  
      Type: `GRB.INTEGER` (non‑negative integers).  

5.  **Identify Parameters (from Schema):**  
    - `Capacity[i]` – capacity of display area *i* (from `capacity.csv`).  
    - `Value[j]` – monetary value of one unit of boat type *j* (from `products.csv`).  
    - `Weight[j]` – physical size (weight) of one unit of boat type *j* (from `products.csv`).  

6.  **Formulate Objective:**  
    Maximize total value of boats displayed:  
    \[
    \max \sum_{i \in I} \sum_{j \in J} \text{Value}[j] \times x[i,j]
    \]

7.  **Formulate Constraints:**  
    - **Capacity Constraints (one per display area):**  
      \[
      \sum_{j \in J} \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i] \quad \forall i \in I
      \]
    - **Non‑negativity and Integer Constraints:**  
      \[
      x[i,j] \;\ge\; 0 \quad \text{and integer} \quad \forall i \in I,\, j \in J
      \]
[Abstract Model Plan END]