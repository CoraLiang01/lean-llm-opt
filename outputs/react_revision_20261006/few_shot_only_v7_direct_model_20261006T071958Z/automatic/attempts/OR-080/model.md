[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect operational constraints: minimum up/down times, ramping limits, and truck capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from truck_id in parameters.csv)
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck \(i\) is started up (turned on) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (from parameters.csv, per truck).
    -   Startup cost: `S` (from parameters.csv, per truck).
    -   Unit transportation cost: `C` (from parameters.csv, per truck).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from parameters.csv, but all trucks have the same values; use these as period demands).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i,t} S_i \cdot z[i,t] \)
    - Transportation costs: \( \sum_{i,t} C_i \cdot x[i,t] \)
    - Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S_i \cdot z[i,t] + C_i \cdot x[i,t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \) for \( t=1,2,3,4 \)
    -   **Spare-Capacity Buffer (Upper Bound):** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \cdot \sum_{i=1}^{10} Q_i \cdot y[i,t] \) for all \( t \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q_i \cdot y[i,t] \) for all \( i, t \)
        - \( x[i,t] \geq 0 \) for all \( i, t \)
    -   **Activation/Startup Logic:** For each truck and period, startup variable is 1 if truck is turned on in period \(t\) (i.e., was off in \(t-1\) and on in \(t\)):
        - For \( t=1 \): \( z[i,1] = y[i,1] \) (since all trucks are initially off)
        - For \( t>1 \): \( z[i,t] \geq y[i,t] - y[i,t-1] \), and \( z[i,t] \leq y[i,t] \), \( z[i,t] \leq 1 - y[i,t-1] \)
    -   **Minimum Up-Time (at least 2 consecutive periods on after startup):**
        - If \( z[i,t]=1 \), then \( y[i,t+1]=1 \) (for \( t=1,2,3 \); cannot start in period 4)
        - Enforce \( z[i,4]=0 \) (no startup allowed in period 4)
    -   **Minimum Down-Time (after shutdown, must stay off for 2 periods):**
        - If truck is turned off at \(t\) (i.e., \( y[i,t-1]=1, y[i,t]=0 \)), then \( y[i,t+1]=0 \) and \( y[i,t+2]=0 \) (for \( t=1,2 \); for \( t=3 \), only \( y[i,4]=0 \))
        - This can be encoded by: \( y[i,t-1] - y[i,t] \leq 1 - y[i,t+1] \) and \( y[i,t-1] - y[i,t] \leq 1 - y[i,t+2] \) where applicable.
    -   **No Startup in Period 4:** \( z[i,4]=0 \) for all \( i \)
    -   **Ramping Constraints (change in transported weight per truck between adjacent periods ≤ 300 kg):**
        - For \( t=2,3,4 \): \( |x[i,t] - x[i,t-1]| \leq 300 \) for all \( i \)
        - This is implemented as two inequalities: \( x[i,t] - x[i,t-1] \leq 300 \) and \( x[i,t-1] - x[i,t] \leq 300 \)
    -   **Inactive Truck Must Transport Zero:** \( x[i,t]=0 \) whenever \( y[i,t]=0 \) (already enforced by \( x[i,t] \leq Q_i \cdot y[i,t] \))
    -   **Initial State:** All trucks are off before period 1 (\( y[i,0]=0 \)), and \( x[i,0]=0 \) for ramping constraints.
[Abstract Model Plan END]