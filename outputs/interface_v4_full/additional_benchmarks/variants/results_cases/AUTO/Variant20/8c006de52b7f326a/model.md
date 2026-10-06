[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost subset of sensor sites to install, such that every monitoring zone is covered by at least one installed sensor. Each candidate sensor site has a cost and covers a specific subset of zones.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a classic Set Covering Problem.
3.  **Define Index Sets:** The primary indices are:
    - Sensor Sites (from sensor_sites.csv, column 'Center')
    - Monitoring Zones (from monitoring_zones.csv, column 'Zone')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if a sensor is installed at site i, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'OpeningCost' from sensor_sites.csv (cost to install at each site).
    -   Coverage mapping: 'CoveredDistricts' from sensor_sites.csv (which zones each site covers).
    -   Set of all zones to be covered: 'Zone' from monitoring_zones.csv.
6.  **Formulate Objective:** Minimize the total installation cost, i.e., minimize the sum over all sensor sites of (OpeningCost[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each monitoring zone j, the sum over all sensor sites i that cover zone j of y[i] must be at least 1. (This ensures every zone is covered by at least one installed sensor.)
    -   Binary Restriction: For all sensor sites i, y[i] ∈ {0,1}.
[Abstract Model Plan END]