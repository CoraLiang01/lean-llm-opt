[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost subset of sensor sites to install, such that every monitoring zone is covered by at least one installed sensor. Each candidate site has a cost and covers a specified set of zones.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) set covering problem.
3.  **Define Index Sets:** The primary indices are:
    - Sensor Sites (indexed by i, from 'sensor_sites.csv', column 'Center')
    - Monitoring Zones (indexed by j, from 'monitoring_zones.csv', column 'Zone')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if a sensor is installed at site i, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Installation costs from 'sensor_sites.csv', column 'OpeningCost' for each site i.
    -   Constraint coefficients: Coverage mapping from 'sensor_sites.csv', column 'CoveredDistricts' (which zones each site covers).
    -   Constraint RHS: Each monitoring zone j (from 'monitoring_zones.csv') must be covered at least once (RHS = 1).
6.  **Formulate Objective:** Minimize the total installation cost, i.e., sum over all sites i of ('OpeningCost'[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each monitoring zone j, the sum over all sites i that cover zone j of y[i] must be at least 1 (i.e., every zone is covered by at least one selected site).
    -   Binary Restriction: For all sites i, y[i] ∈ {0,1}.
[Abstract Model Plan END]