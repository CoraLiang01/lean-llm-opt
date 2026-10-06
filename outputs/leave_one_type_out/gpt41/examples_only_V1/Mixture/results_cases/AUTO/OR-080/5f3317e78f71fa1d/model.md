[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and continuous transportation variables, subject to scheduling and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, 2, ..., 10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `u[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (kg) per truck.
    -   Startup cost: `S` per truck.
    -   Unit transportation cost: `C` per truck.
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (identical for all trucks, so use once per period).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i, t} S[i] \cdot u[i, t] \)
    - Transportation costs: \( \sum_{i, t} C[i] \cdot x[i, t] \)
    - So, Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S[i] \cdot u[i, t] + C[i] \cdot x[i, t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), total transported weight must meet or exceed customer demand:
        - \( \sum_{i=1}^{10} x[i, t] \geq d_t \) (where \(d_t\) is from `d1`–`d4`)
    -   **Spare-Capacity Buffer:** For each period \(t\), total transported weight cannot exceed 90% of the sum of active trucks' capacities:
        - \( \sum_{i=1}^{10} x[i, t] \leq 0.9 \cdot \sum_{i=1}^{10} Q[i] \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    -   **Startup Logic:** For each truck and period, startup variable is 1 if truck is turned on in period \(t\) after being off in \(t-1\):
        - For \(t=1\): \( u[i, 1] = y[i, 1] \) (since all trucks are initially off)
        - For \(t>1\): \( u[i, t] \geq y[i, t] - y[i, t-1] \)
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. So, for any startup at period \(t\), enforce \( y[i, t+1] \geq u[i, t] \) for \( t=1,2,3 \) (cannot start in period 4).
        - Additionally, prohibit startups in period 4: \( u[i, 4] = 0 \)
    -   **Minimum Down-Time:** If a truck is shut down (i.e., \( y[i, t-1]=1, y[i, t]=0 \)), it must remain off in periods \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For \( t=2,3 \): \( y[i, t-1] - y[i, t] \leq 1 - y[i, t+1] \) (if off at \(t\), must be off at \(t+1\))
        - For \( t=3 \): If off at \(t=3\), must be off at \(t=4\) as well.
    -   **Ramping (Load Change) Constraint:** For each truck and consecutive periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - For \( t=2,3,4 \): \( |x[i, t] - x[i, t-1]| \leq 300 \)
    -   **Inactive Truck Must Not Transport:** For all \(i, t\): If \( y[i, t]=0 \), then \( x[i, t]=0 \) (already enforced by \( x[i, t] \leq Q[i] \cdot y[i, t] \))
    -   **Initial State:** All trucks are off before period 1: \( y[i, 0]=0 \) (for logic in startup and ramping constraints).
    -   **No Startup in Last Period:** \( u[i, 4]=0 \) (cannot start in period 4, as minimum up-time cannot be satisfied).
[Abstract Model Plan END]