[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and satisfy operational constraints including minimum up/down times, ramping limits, and truck capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from parameters.csv, all rows)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    - \( x_{i,t} \) = Amount of weight (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x_{i,t} \geq 0 \).
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \) = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q_i \) (column 'Q')
    - Startup cost: \( S_i \) (column 'S')
    - Unit transportation cost: \( C_i \) (column 'C')
    - Period demands: \( d_t \) (columns 'd1', 'd2', 'd3', 'd4')
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of startup costs (incurred when a truck is started) and transportation costs (proportional to weight transported):
    - \( \text{Minimize} \sum_{i} \sum_{t} S_i \cdot s_{i,t} + \sum_{i} \sum_{t} C_i \cdot x_{i,t} \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), total transported weight must be at least the customer demand:
        - \( \sum_{i} x_{i,t} \geq d_t \)
    - **Spare-Capacity Buffer:** For each period \( t \), total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \)
    - **Truck Capacity:** For each truck \( i \) and period \( t \), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( 0 \leq x_{i,t} \leq Q_i \cdot y_{i,t} \)
    - **Startup Logic:** For each truck \( i \) and period \( t \), startup occurs if truck is on in \( t \) but was off in \( t-1 \) (with all trucks off at \( t=0 \)):
        - \( s_{i,1} = y_{i,1} \)
        - For \( t > 1 \): \( s_{i,t} \geq y_{i,t} - y_{i,t-1} \), \( s_{i,t} \geq 0 \)
    - **Minimum Up-Time:** If a truck is started in period \( t \), it must remain active in period \( t \) and \( t+1 \) (cannot start in period 4):
        - For \( t \in \{1,2,3\} \): \( y_{i,t} + y_{i,t+1} \geq 2 \cdot s_{i,t} \)
        - \( s_{i,4} = 0 \)
    - **Minimum Down-Time:** If a truck is shut down in period \( t \) (i.e., \( y_{i,t-1}=1, y_{i,t}=0 \)), it must remain off in periods \( t \) and \( t+1 \), and cannot restart before \( t+2 \):
        - For \( t \in \{1,2,3\} \): \( y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \)
        - For \( t=4 \), no further constraint needed.
    - **Ramping (Change in Load):** For each truck \( i \) and periods \( t=2,3,4 \), the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):
        - \( |x_{i,t} - x_{i,t-1}| \leq 300 \)
    - **Initial Conditions:** All trucks are off before period 1 (\( y_{i,0} = 0 \)), and \( x_{i,0} = 0 \) for ramping constraints.
[Abstract Model Plan END]