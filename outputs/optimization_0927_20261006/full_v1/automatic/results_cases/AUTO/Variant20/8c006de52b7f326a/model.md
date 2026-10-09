[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost subset of sensor sites to install monitoring sensors such that every monitoring zone is covered by at least one installed sensor. Each candidate site has a cost and covers a specified subset of zones.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) set covering problem.
3.  **Define Index Sets:** The primary indices are:
    - Sensor Sites (from sensor_sites.csv, column 'Center')
    - Monitoring Zones (from monitoring_zones.csv, column 'Zone')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if a sensor is installed at site i, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Installation costs from 'OpeningCost' in sensor_sites.csv.
    -   Coverage mapping: For each site i, the set of zones it covers from 'CoveredDistricts' in sensor_sites.csv.
    -   Constraint RHS: Each monitoring zone (from monitoring_zones.csv) must be covered at least once (RHS = 1).
6.  **Formulate Objective:** Minimize the total installation cost, i.e., minimize the sum over all sites of ('OpeningCost' for site i) × y[i].
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each monitoring zone j, the sum of y[i] over all sites i that cover zone j must be at least 1 (i.e., every zone is covered by at least one selected site).
    -   Binary Restriction: For all sites i, y[i] ∈ {0,1}.
[Abstract Model Plan END]