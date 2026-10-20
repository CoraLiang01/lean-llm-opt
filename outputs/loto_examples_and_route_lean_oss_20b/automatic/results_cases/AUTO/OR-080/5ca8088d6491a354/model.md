##### Decision Variables

$y_{it}\in\{0,1\}$: binary variable equal to 1 if truck $i$ is active in period $t$, 0 otherwise, for $i=1,\dots ,10$ and $t=1,\dots ,4$.

$s_{it}\in\{0,1\}$: binary variable equal to 1 if truck $i$ is started in period $t$, 0 otherwise, for $i=1,\dots ,10$ and $t=1,\dots ,3$ (no start in period 4).

$w_{it}\geq0$: continuous variable equal to the weight (in kg) transported by truck $i$ in period $t$, for $i=1,\dots ,10$ and $t=1,\dots ,4$.

##### Objective Function

$\displaystyle \min\;\sum_{i=1}^{10}\Bigl(S_i\bigl(s_{i1}+s_{i2}+s_{i3}\bigr)+\sum_{t=1}^{4}C_i\,w_{it}\Bigr)$

##### Constraints

1. **Startup definition**  
   $\displaystyle s_{i1}\leq y_{i1}\quad\forall i$  
   $\displaystyle s_{i2}\leq y_{i2}\quad\forall i$  
   $\displaystyle s_{i3}\leq y_{i3}\quad\forall i$  

   $\displaystyle s_{i2}\geq y_{i2}-y_{i1}\quad\forall i$  
   $\displaystyle s_{i3}\geq y_{i3}-y_{i2}\quad\forall i$  

   $\displaystyle s_{i1}\geq y_{i1}\quad\forall i$  
   $\displaystyle s_{i2}\geq y_{i2}-y_{i1}\quad\forall i$  
   $\displaystyle s_{i3}\geq y_{i3}-y_{i2}\quad\forall i$  

2. **At least two consecutive active periods**  
   $\displaystyle y_{i1}\leq y_{i2}\quad\forall i$  
   $\displaystyle y_{i2}\leq y_{i3}\quad\forall i$  
   $\displaystyle y_{i3}\leq y_{i4}\quad\forall i$  

3. **No start in period 4**  
   $\displaystyle y_{i4}\leq y_{i3}\quad\forall i$  

4. **Weight capacity**  
   $\displaystyle w_{it}\leq Q_i\,y_{it}\quad\forall i,\;t$  

5. **Weight change limit**  
   $\displaystyle w_{i1}\leq 300\quad\forall i$  
   $\displaystyle w_{i1}\geq 0\quad\forall i$  

   $\displaystyle w_{it}-w_{i(t-1)}\leq 300\quad\forall i,\;t=2,3,4$  
   $\displaystyle w_{i(t-1)}-w_{it}\leq 300\quad\forall i,\;t=2,3,4$  

6. **Demand satisfaction**  
   $\displaystyle \sum_{i=1}^{10}w_{it}\geq d_t\quad\forall t=1,2,3,4$  

7. **Spare‑capacity buffer**  
   $\displaystyle \sum_{i=1}^{10}w_{it}\leq 0.9\sum_{i=1}^{10}Q_i\,y_{it}\quad\forall t=1,2,3,4$  

8. **Variable domains**  
   $y_{it}\in\{0,1\},\;s_{it}\in\{0,1\},\;w_{it}\geq0\quad\forall i,t$  

##### Retrieved Information

{
  "trucks": [
    {
      "truck_id": "1",
      "Q": 1000,
      "S": 500,
      "C": 2.0,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "2",
      "Q": 800,
      "S": 300,
      "C": 3.0,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "3",
      "Q": 1200,
      "S": 400,
      "C": 2.5,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "4",
      "Q": 600,
      "S": 250,
      "C": 3.0,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "5",
      "Q": 900,
      "S": 450,
      "C": 2.2,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "6",
      "Q": 700,
      "S": 280,
      "C": 2.8,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "7",
      "Q": 1100,
      "S": 420,
      "C": 2.4,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "8",
      "Q": 500,
      "S": 200,
      "C": 3.2,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "9",
      "Q": 1000,
      "S": 480,
      "C": 2.1,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    },
    {
      "truck_id": "10",
      "Q": 650,
      "S": 260,
      "C": 2.9,
      "d1": 1500,
      "d2": 2000,
      "d3": 1800,
      "d4": 1000
    }
  ]
}