[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost subset of sensor sites to install, such that every monitoring zone is covered by at least one installed sensor. Each candidate sensor site has a known installation cost and covers a specified subset of monitoring zones.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) set covering problem.
3.  **Define Index Sets:** The primary indices are:
    - Sensor Sites (indexed by i, from 'sensor_sites.csv', column 'Center')
    - Monitoring Zones (indexed by j, from 'monitoring_zones.csv', column 'Zone')
4.  **Define Decision Variables:**
    - `y[i]` = 1 if a sensor is installed at site i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: Installation costs from 'sensor_sites.csv', column 'OpeningCost'.
    - Coverage mapping: For each site i, the set of zones it covers from 'sensor_sites.csv', column 'CoveredDistricts' (parsed as a list).
    - The complete set of monitoring zones from 'monitoring_zones.csv', column 'Zone'.
6.  **Formulate Objective:** Minimize the total installation cost, i.e., minimize the sum over all sensor sites of ('OpeningCost' for site i) × y[i].
7.  **Formulate Constraints:**
    - Coverage Constraint: For each monitoring zone j, the sum over all sensor sites i that cover zone j of y[i] must be at least 1 (i.e., every zone must be covered by at least one selected sensor site).
    - Binary Restriction: For each sensor site i, y[i] ∈ {0,1}.
[Abstract Model Plan END]