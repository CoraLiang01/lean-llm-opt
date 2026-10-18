[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items from a given list so that the total value is maximized while the total weight does not exceed the available capacity of 15 units. Each item can be chosen at most once, so the decision variables are binary.
2.  **Identify Model Type:** Based on the query, this is a **0‑1 Knapsack** (a binary integer programming) problem.
3.  **Define Index Sets:** The primary index set is  
   - `I = {1, 2, …, 140}`  (the 140 items listed in `value.csv`).
4.  **Define Decision Variables:**
    - `x[i]` = 1 if item *i* is selected, 0 otherwise. Type: **GRB.BINARY**.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: `value[i]` (column “value” in `value.csv`).
    - Constraint coefficients: `weight[i]` (column “weight” in `value.csv`).
    - Constraint RHS: the weight capacity `W = 15`.
6.  **Formulate Objective:**  
   Maximize  
   \[
   \sum_{i \in I} \text{value}[i] \cdot x[i]
   \]
7.  **Formulate Constraints:**
    - **Weight Capacity:**  
      \[
      \sum_{i \in I} \text{weight}[i] \cdot x[i] \;\le\; 15
      \]
    - **Binary Decision Variables:**  
      \[
      x[i] \in \{0,1\} \quad \forall i \in I
      \]
[Abstract Model Plan END]