[Abstract Model Plan START]
1. **Analyze Query:** The user wants to determine the optimal visiting sequence for a single delivery van that starts and ends at the depot, visits customers A, B, and C exactly once, and minimizes the total travel distance.
2. **Identify Model Type:** This is a Mixed Integer Programming (MIP) formulation of the Traveling Salesman Problem (TSP).
3. **Define Index Sets:**  
   - `Nodes = {Depot, A, B, C}`  
   - `Pairs = {(i,j) | i ∈ Nodes, j ∈ Nodes, i ≠ j}`
4. **Define Decision Variables:**  
   - `x[i,j]` = 1 if the van travels directly from node *i* to node *j*, 0 otherwise. **Type:** `GRB.BINARY`.  
   - `u[i]` = integer position of node *i* in the tour (used for subtour elimination). For `i ∈ {A,B,C}`. **Type:** `GRB.INTEGER`, bounds `1 ≤ u[i] ≤ 3`.  
   - `u[Depot]` is fixed to 0 (constant, not a decision variable).
5. **Identify Parameters (from Schema):**  
   - `d[i,j]` = road distance (km) between node *i* and node *j*, extracted from `DistanceMatrix.csv` for all pairs in `Pairs`.  
   - The matrix is symmetric, so `d[i,j] = d[j,i]`, but we treat directed edges for the formulation.
6. **Formulate Objective:**  
   Minimize total travel distance  
   \[
   \min \sum_{i \in \text{Nodes}} \sum_{\substack{j \in \text{Nodes}\\ j \neq i}} d[i,j] \cdot x[i,j]
   \]
7. **Formulate Constraints:**  
   - **Outgoing constraint (each node leaves once):**  
     \[
     \sum_{\substack{j \in \text{Nodes}\\ j \neq i}} x[i,j] = 1 \quad \forall i \in \text{Nodes}
     \]
   - **Incoming constraint (each node is entered once):**  
     \[
     \sum_{\substack{j \in \text{Nodes}\\ j \neq i}} x[j,i] = 1 \quad \forall i \in \text{Nodes}
     \]
   - **Subtour elimination (MTZ formulation):**  
     For all `i, j ∈ {A,B,C}`, `i ≠ j`:  
     \[
     u[i] - u[j] + 3 \cdot x[i,j] \le 2
     \]
     (Here `|Nodes| - 1 = 3` and `u[Depot] = 0` is fixed.)
   - **Bounds on `u` variables:**  
     \[
     1 \le u[i] \le 3 \quad \forall i \in \{A,B,C\}
     \]
   - **Variable types:** `x[i,j] ∈ {0,1}`; `u[i]` integer as specified.

[Abstract Model Plan END]