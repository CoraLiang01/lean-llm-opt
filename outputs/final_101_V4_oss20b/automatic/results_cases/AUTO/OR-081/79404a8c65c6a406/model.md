[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one‑day meal plan that uses the available foods listed in *cost.csv* to **minimize the total cost** while satisfying specific nutritional requirements: at least 2000 kcal, at least 50 g protein, at least 60 mg vitamin C, and no more than 70 g fat. Portions may be fractional.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem with continuous decision variables and linear objective/constraints.

3.  **Define Index Sets:**  
    - `F` : set of all foods (120 items) indexed by `i`.

4.  **Define Decision Variables:**  
    - `x[i]` = number of servings of food `i` (continuous, non‑negative).  
      Type: `GRB.CONTINUOUS`.  
      Domain: `x[i] ≥ 0`.

5.  **Identify Parameters (from Schema):**  
    - **Objective coefficients**: `Cost[i]` (USD per serving).  
    - **Constraint coefficients**:  
        - `Calories[i]` (kcal per serving).  
        - `Protein[i]` (g per serving).  
        - `Fat[i]` (g per serving).  
        - `VitaminC[i]` (mg per serving).  
    - **Constraint RHS (limits)**:  
        - Minimum calories: `2000`.  
        - Minimum protein: `50`.  
        - Minimum vitamin C: `60`.  
        - Maximum fat: `70`.

6.  **Formulate Objective:**  
    Minimize total cost  
    \[
    \min \sum_{i \in F} \text{Cost}[i] \cdot x[i]
    \]

7.  **Formulate Constraints:**  
    - **Calories**:  
      \[
      \sum_{i \in F} \text{Calories}[i] \cdot x[i] \;\ge\; 2000
      \]
    - **Protein**:  
      \[
      \sum_{i \in F} \text{Protein}[i] \cdot x[i] \;\ge\; 50
      \]
    - **Vitamin C**:  
      \[
      \sum_{i \in F} \text{VitaminC}[i] \cdot x[i] \;\ge\; 60
      \]
    - **Fat**:  
      \[
      \sum_{i \in F} \text{Fat}[i] \cdot x[i] \;\le\; 70
      \]
    - **Non‑negativity**:  
      \[
      x[i] \;\ge\; 0 \quad \forall i \in F
      \]
[Abstract Model Plan END]