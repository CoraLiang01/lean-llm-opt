[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three specific customer locations (A, B, C) exactly once each, starting and ending at the depot, so as to minimize the total travel distance. The pairwise distances between all locations (including the depot) are provided in DistanceMatrix.csv. No other locations are to be visited.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a Mixed-Integer Programming (MIP) problem.
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}.
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY, for all ordered pairs (i, j) where i ≠ j and i, j ∈ {Depot, A, B, C}.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: The travel distances between each pair of locations, from the relevant cells in DistanceMatrix.csv (columns and rows: 'Depot', 'A', 'B', 'C').
    -   No other parameters are needed, as there are no time windows, capacities, or service times.
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of distance[i][j] * x[i,j], where distance[i][j] is taken from DistanceMatrix.csv for i ≠ j and i, j ∈ {Depot, A, B, C}.
7.  **Formulate Constraints:**
    -   Constraint 1 (Visit Each Location Once): For each location k ∈ {A, B, C}, the sum over all i ≠ k of x[i,k] = 1 (each location is entered exactly once), and the sum over all j ≠ k of x[k,j] = 1 (each location is exited exactly once).
    -   Constraint 2 (Depot Start and End): The depot must be departed from exactly once and returned to exactly once: sum over j ≠ Depot of x[Depot, j] = 1, and sum over i ≠ Depot of x[i, Depot] = 1.
    -   Constraint 3 (Subtour Elimination): To prevent disconnected subtours, add subtour elimination constraints (e.g., Miller-Tucker-Zemlin or explicit enumeration for this small case).
    -   No other constraints are needed, as there are no additional requirements.
[Abstract Model Plan END]