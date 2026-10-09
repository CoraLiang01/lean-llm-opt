[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to minimize total startup and transportation costs, while satisfying customer demand, truck operational constraints (minimum up/down times), ramping limits, and spare-capacity requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from parameters.csv, all rows used)
    - Periods: \( t \in \{1,2,3,4\} \) (explicitly enumerated in the query)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (from parameters.csv, per truck)
    -   Startup cost: `S` (from parameters.csv, per truck)
    -   Unit transportation cost: `C` (from parameters.csv, per truck)
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from parameters.csv, but all trucks have the same values; use these as period demands)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \( S[i] \times z[i,t] \) (incurred when a truck is started in period \( t \))
    -   Transportation costs: \( C[i] \times x[i,t] \)
    -   Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S[i] \cdot z[i,t] + C[i] \cdot x[i,t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \( t \), the total transported weight must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \) (where \( d_t \) is the demand for period \( t \))
    -   **Spare-Capacity Buffer:** For each period \( t \), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \times \sum_{i=1}^{10} Q[i] \cdot y[i,t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q[i] \cdot y[i,t] \)
        - \( x[i,t] \geq 0 \)
    -   **Startup Logic:** For each truck and period, startup variable is 1 if truck is turned on in period \( t \) after being off in \( t-1 \):
        - For \( t=1 \): \( z[i,1] = y[i,1] \) (since all trucks are initially off)
        - For \( t>1 \): \( z[i,t] \geq y[i,t] - y[i,t-1] \), and \( z[i,t] \leq 1 \)
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. Therefore, for any startup in period \( t \) (where \( t \leq 3 \)), require \( y[i,t+1] \geq z[i,t] \). No startup allowed in period 4: \( z[i,4] = 0 \).
    -   **Minimum Down-Time:** If a truck is shut down in period \( t \) (i.e., \( y[i,t-1]=1, y[i,t]=0 \)), it must remain off in periods \( t \) and \( t+1 \), and cannot be restarted before \( t+2 \). For \( t=2,3 \): If \( y[i,t-1]=1, y[i,t]=0 \), then \( y[i,t+1]=0 \). (Implement via constraints: \( y[i,t-1] - y[i,t] \leq 1 - y[i,t+1] \) or equivalent logic.)
    -   **No Startup in Period 4:** \( z[i,4] = 0 \) for all \( i \).
    -   **Ramping (Change) Constraints:** For each truck and consecutive periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - For \( t=2,3,4 \): \( |x[i,t] - x[i,t-1]| \leq 300 \)
    -   **Inactive Truck Must Transport Zero:** For all \( i, t \): \( x[i,t] \leq Q[i] \cdot y[i,t] \), and \( x[i,t] \geq 0 \).
    -   **Initial State:** All trucks are off before period 1 (\( y[i,0]=0 \)), and no initial transported weight is defined (no ramp constraint for \( t=1 \)).
[Abstract Model Plan END]