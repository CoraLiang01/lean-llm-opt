[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three specific customer locations (A, B, C) exactly once each, starting and ending at the depot, so as to minimize the total travel distance. The pairwise distances between all relevant locations are provided in DistanceMatrix.csv. No other locations are to be considered.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a Mixed-Integer Programming (MIP) problem.
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}.
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY, for all ordered pairs (i, j) where i ≠ j and i, j ∈ {Depot, A, B, C}.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: The travel distances between each pair of locations, from the corresponding entries in DistanceMatrix.csv (e.g., DistanceMatrix[i][j]).
    -   Constraint coefficients: The same distance matrix is used to define feasible arcs and to ensure each location is entered and exited exactly once.
    -   Constraint RHS: Each location (including the depot) must be departed from and arrived at exactly once.
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of DistanceMatrix[i][j] * x[i,j], where x[i,j] = 1 if the van travels directly from i to j.
7.  **Formulate Constraints:**
    -   Constraint 1 (Visit Each Location Once): For each location (including the depot), the sum of outgoing arcs x[i,j] over all j ≠ i equals 1 (the van leaves each location exactly once).
    -   Constraint 2 (Arrive at Each Location Once): For each location (including the depot), the sum of incoming arcs x[j,i] over all j ≠ i equals 1 (the van arrives at each location exactly once).
    -   Constraint 3 (Subtour Elimination): To prevent disconnected cycles (subtours), add constraints (e.g., Miller-Tucker-Zemlin or explicit enumeration for small problems) to ensure the route forms a single tour visiting all locations.
[Abstract Model Plan END]