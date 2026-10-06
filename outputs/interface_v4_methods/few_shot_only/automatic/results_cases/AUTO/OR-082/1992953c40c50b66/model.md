[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three customer locations (A, B, C) exactly once each, starting and ending at the depot, so that the total travel distance is minimized. The pairwise distances between all locations (including the depot) are provided in DistanceMatrix.csv.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a combinatorial optimization problem typically formulated as a Mixed Integer Program (MIP).
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}.
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Distance between locations i and j will come from the DistanceMatrix.csv, specifically from the intersection of row i and column j (columns: 'Depot', 'A', 'B', 'C'; rows: 'Depot', 'A', 'B', 'C').
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of distance[i][j] * x[i,j], where the sum is over all ordered pairs of distinct locations in {Depot, A, B, C}.
7.  **Formulate Constraints:**
    -   Constraint 1 (Visit Each Location Once): For each location (including the depot), the number of routes entering the location equals the number of routes leaving the location, and for customer locations (A, B, C), exactly one route enters and one leaves.
    -   Constraint 2 (Start and End at Depot): The route must start and end at the depot (i.e., exactly one route leaves the depot and one returns to it).
    -   Constraint 3 (Subtour Elimination): Prevent the formation of disconnected cycles (subtours) that do not include the depot, ensuring a single tour that visits all locations.
[Abstract Model Plan END]