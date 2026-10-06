[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost subset of sensor sites to install, such that every monitoring zone is covered by at least one installed sensor. Each candidate sensor site has an installation cost and covers a specified subset of monitoring zones.
2.  **Identify Model Type:** Based on the query, this is a classic Minimum Set Covering problem, formulated as a Mixed-Integer Programming (MIP) model.
3.  **Define Index Sets:** The primary indices are:
    - Sensor Sites (from sensor_sites.csv, column 'Center')
    - Monitoring Zones (from monitoring_zones.csv, column 'Zone')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if a sensor is installed at site i, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Installation costs from 'OpeningCost' column in sensor_sites.csv, indexed by sensor site.
    -   Coverage mapping: For each sensor site (from 'CoveredDistricts' in sensor_sites.csv), the set of monitoring zones it covers.
    -   The complete set of monitoring zones to be covered is from monitoring_zones.csv, column 'Zone'.
6.  **Formulate Objective:** Minimize the total installation cost, i.e., the sum over all sensor sites of (OpeningCost[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each monitoring zone j (from monitoring_zones.csv), the sum over all sensor sites i that cover zone j of y[i] must be at least 1. This ensures every zone is covered by at least one installed sensor.
    -   Binary Restriction: For each sensor site i, y[i] ∈ {0,1}.
[Abstract Model Plan END]