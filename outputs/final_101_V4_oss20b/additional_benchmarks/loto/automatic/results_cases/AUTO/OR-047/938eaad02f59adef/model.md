[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide how many units of each game genre to list on each platform so that the total value of all listed games is maximised, while ensuring that the total memory used on each platform does not exceed that platform’s capacity.  
2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem: linear objective, linear constraints, and integer decision variables.  
3.  **Define Index Sets:**  
    - **Platforms** \(I = \{1,2,\dots,10\}\) (from *capacity.csv*).  
    - **Products / Genres** \(J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}\}\) (from *products.csv*).  
4.  **Define Decision Variables:**  
    - `x[i,j]` = number of units of product \(j\) listed on platform \(i\).  
      Type: **GRB.INTEGER** (non‑negative).  
5.  **Identify Parameters (from Schema):**  
    - **Value** \(v_j\) – the predefined value of product \(j\) (column *Value* in *products.csv*).  
    - **Weight** \(w_j\) – the memory requirement of product \(j\) (column *Weight* in *products.csv*).  
    - **Capacity** \(c_i\) – the memory capacity of platform \(i\) (column *Capacity* in *capacity.csv*).  
6.  **Formulate Objective:**  
    \[
    \text{Maximise } \sum_{i \in I}\;\sum_{j \in J}\; v_j \; x_{i,j}
    \]  
    (total value of all listed games).  
7.  **Formulate Constraints:**  
    - **Capacity constraints (one per platform):**  
      \[
      \sum_{j \in J} w_j \; x_{i,j} \;\le\; c_i \quad \forall i \in I
      \]  
      (total memory used on platform \(i\) cannot exceed its capacity).  
    - **Non‑negativity and integrality:**  
      \[
      x_{i,j} \;\ge\; 0 \quad \text{and integer} \quad \forall i \in I,\; j \in J
      \]  
    - (No other constraints are required; the model automatically respects the capacity limits and integer nature of the variables.)  
[Abstract Model Plan END]