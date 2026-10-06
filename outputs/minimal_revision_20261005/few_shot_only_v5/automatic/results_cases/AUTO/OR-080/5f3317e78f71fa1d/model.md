[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect truck-specific startup, minimum up/down time, ramping, and capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and scheduling (minimum up/down time, ramping) features.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from truck_id in parameters.csv)
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck \(i\) is started up (turned on from off) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (from parameters.csv, per truck)
    -   Startup cost: `S` (from parameters.csv, per truck)
    -   Unit transportation cost: `C` (from parameters.csv, per truck)
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from parameters.csv, but all trucks have the same values; use once per period)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i,t} S_i \cdot z[i,t] \)
    - Transportation costs: \( \sum_{i,t} C_i \cdot x[i,t] \)
    So, Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S_i \cdot z[i,t] + C_i \cdot x[i,t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \) for \( t=1,2,3,4 \)
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \cdot \sum_{i=1}^{10} Q_i \cdot y[i,t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q_i \cdot y[i,t] \) for all \(i, t\)
        - \( x[i,t] \geq 0 \)
    -   **Startup Logic:** For each truck and period, startup variable is 1 if truck is turned on from off:
        - For \( t=1 \): \( z[i,1] \geq y[i,1] \) (since all trucks are initially off)
        - For \( t>1 \): \( z[i,t] \geq y[i,t] - y[i,t-1] \)
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. So, if \( z[i,t]=1 \), then \( y[i,t+1]=1 \) (for \( t=1,2,3 \); cannot start in period 4):
        - \( z[i,t] \leq y[i,t+1] \) for \( t=1,2,3 \)
        - \( z[i,4]=0 \) (cannot start in period 4)
    -   **Minimum Down-Time:** If a truck is shut down (on in \(t-1\), off in \(t\)), it must remain off in \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For \( t=1,2,3 \): If \( y[i,t-1]=1 \) and \( y[i,t]=0 \), then \( y[i,t+1]=0 \)
        - This can be enforced by: \( y[i,t-1] - y[i,t] \leq 1 - y[i,t+1] \) for \( t=2,3 \)
        - Additionally, prevent restart before \(t+2\): \( z[i,t+1] \leq 1 - (y[i,t-1] - y[i,t]) \) for \( t=2,3 \)
    -   **Ramping (Change in Transported Weight):** For each truck and consecutive periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - \( |x[i,t] - x[i,t-1]| \leq 300 \) for \( t=2,3,4 \)
        - Implemented as two inequalities:
            - \( x[i,t] - x[i,t-1] \leq 300 \)
            - \( x[i,t-1] - x[i,t] \leq 300 \)
        - For \( t=1 \), no ramp constraint applies since initial state is off and transported weight is zero.
    -   **No Transport When Inactive:** \( x[i,t]=0 \) whenever \( y[i,t]=0 \) (already enforced by capacity constraint).
    -   **Variable Domains:** \( y[i,t}, z[i,t] \in \{0,1\} \), \( x[i,t] \geq 0 \).
[Abstract Model Plan END]