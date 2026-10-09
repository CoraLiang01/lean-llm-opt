[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demands, minimize total startup and transportation costs, and satisfy operational constraints including minimum up/down times, ramping limits, and spare-capacity requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck \(i\) is started up (turned on) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Maximum truck capacity: `Q` (per truck, from parameters.csv)
    -   Startup cost: `S` (per truck, from parameters.csv)
    -   Unit transportation cost: `C` (per truck, from parameters.csv)
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from parameters.csv; all trucks have the same values)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \( \sum_{i,t} S[i] \cdot z[i,t] \)
    -   Transportation costs: \( \sum_{i,t} C[i] \cdot x[i,t] \)
    So, Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} [S[i] \cdot z[i,t} + C[i] \cdot x[i,t}] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \)  (where \(d_t\) is the demand in period \(t\))
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \cdot \sum_{i=1}^{10} Q[i] \cdot y[i,t] \)
    -   **Truck Capacity:** For each truck \(i\) and period \(t\), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q[i] \cdot y[i,t] \)
        - \( x[i,t] \geq 0 \)
    -   **Startup Logic:** For each truck \(i\) and period \(t\), define startup variable:
        - For \(t=1\): \( z[i,1] \geq y[i,1] \) (since all trucks are initially off)
        - For \(t>1\): \( z[i,t] \geq y[i,t] - y[i,t-1] \) (startup occurs if truck is off in \(t-1\) and on in \(t\))
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. For each truck \(i\) and period \(t=1,2,3\):
        - If \( z[i,t] = 1 \), then \( y[i,t+1] = 1 \)
        - Equivalently: \( y[i,t+1] \geq z[i,t] \)
        - No startups allowed in period 4: \( z[i,4] = 0 \)
    -   **Minimum Down-Time:** If a truck is shut down (on in \(t-1\), off in \(t\)), it must remain off in \(t\) and \(t+1\), and cannot be restarted before \(t+2\). For each truck \(i\) and period \(t=1,2\):
        - If \( y[i,t] = 1 \) and \( y[i,t+1] = 0 \), then \( y[i,t+2] = 0 \)
        - Equivalently: \( y[i,t] - y[i,t+1] \leq 1 - y[i,t+2] \)
        - For \(t=3\), after shutdown in period 4, no further periods, so no constraint needed.
    -   **Ramping (Change) Constraints:** For each truck \(i\) and periods \(t=1,2,3\):
        - \( |x[i,t+1] - x[i,t]| \leq 300 \)
        - This is implemented as two constraints:
            - \( x[i,t+1] - x[i,t] \leq 300 \)
            - \( x[i,t] - x[i,t+1] \leq 300 \)
        - Note: For periods where a truck is off, \( x[i,t] = 0 \) by the capacity constraint.
    -   **Initial State:** All trucks are off before period 1:
        - \( y[i,0] = 0 \) (for startup logic in period 1)
    -   **No Startup in Period 4:** \( z[i,4] = 0 \) for all \(i\).
    -   **Non-negativity and Binary:** All \( x[i,t] \geq 0 \), \( y[i,t], z[i,t] \in \{0,1\} \).
[Abstract Model Plan END]