[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demands, minimize total startup and transportation costs, and satisfy operational constraints including minimum up/down times, ramping limits, and spare-capacity requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and scheduling features.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, \ldots, 10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    - \( x_{i,t} \) = Amount of weight (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x_{i,t} \geq 0 \).
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \) = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q_i \) (column 'Q')
    - Startup cost: \( S_i \) (column 'S')
    - Unit transportation cost: \( C_i \) (column 'C')
    - Customer demand per period: \( d_t \) (columns 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of startup costs (incurred when a truck is started) and transportation costs (proportional to weight transported):
       \[
       \text{Minimize} \quad \sum_{i} \sum_{t} S_i \cdot s_{i,t} + \sum_{i} \sum_{t} C_i \cdot x_{i,t}
       \]
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), the total transported weight must meet or exceed customer demand:
        \[
        \sum_{i} x_{i,t} \geq d_t
        \]
    - **Spare-Capacity Buffer:** For each period \( t \), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        \[
        \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t}
        \]
    - **Truck Capacity:** For each truck \( i \) and period \( t \), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        \[
        0 \leq x_{i,t} \leq Q_i \cdot y_{i,t}
        \]
    - **Startup Definition:** For each truck \( i \) and period \( t \), startup occurs if the truck is off in \( t-1 \) and on in \( t \):
        \[
        s_{i,1} = y_{i,1} \quad (\text{since all trucks are initially off})
        \]
        \[
        s_{i,t} \geq y_{i,t} - y_{i,t-1} \quad \forall t \geq 2
        \]
        \[
        s_{i,t} \leq 1 - y_{i,t-1} \quad \forall t \geq 2
        \]
        \[
        s_{i,t} \leq y_{i,t} \quad \forall t \geq 2
        \]
    - **Minimum Up-Time:** If a truck is started in period \( t \), it must remain active in period \( t \) and \( t+1 \) (cannot be started in period 4):
        \[
        s_{i,4} = 0
        \]
        \[
        y_{i,t+1} \geq s_{i,t} \quad \forall i, t \in \{1,2,3\}
        \]
    - **Minimum Down-Time:** If a truck is shut down (on in \( t-1 \), off in \( t \)), it must remain off in \( t \) and \( t+1 \), and cannot be restarted before \( t+2 \):
        \[
        y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t} - y_{i,t+1} \quad \forall i, t \in \{2,3\}
        \]
        (Or equivalently, if \( y_{i,t-1}=1, y_{i,t}=0 \), then \( y_{i,t+1}=0 \))
    - **Ramping (Change) Constraints:** For each truck \( i \) and periods \( t=2,3,4 \), the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):
        \[
        |x_{i,t} - x_{i,t-1}| \leq 300
        \]
    - **No Startup in Last Period:** For all trucks, cannot start in period 4:
        \[
        s_{i,4} = 0
        \]
    - **Initial State:** All trucks are off before period 1:
        \[
        y_{i,0} = 0 \quad (\text{for startup logic})
        \]
[Abstract Model Plan END]