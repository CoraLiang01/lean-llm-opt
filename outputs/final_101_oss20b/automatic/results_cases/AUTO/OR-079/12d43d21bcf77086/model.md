##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether factory $i$ is constructed.

##### Objective Function

$\displaystyle \min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i$

##### Constraints

1. **Demand satisfaction**  
   $\displaystyle \sum_{i\in I} x_{ij}=d_j,\quad \forall j\in J$.

2. **Factory activation (upper bound)**  
   $\displaystyle \sum_{j\in J} x_{ij}\leq M\,y_i,\quad \forall i\in I$.

3. **Factory capacity**  
   $\displaystyle \sum_{j\in J} x_{ij}\leq \text{Capacity}_i\,y_i,\quad \forall i\in I$.

4. **Domains**  
   $x_{ij}\geq 0$ (continuous); $y_i\in\{0,1\}$.

$M=\sum_{j\in J} d_j=220$.

$I=\{A1,A2,\dots ,A15\}$, $J=\{B1,B2,\dots ,B8\}$.

##### Retrieved Information

{
  "facilities": [
    {"Facility":"A1","FixedCost":0,"Capacity":30},
    {"Facility":"A2","FixedCost":175,"Capacity":10},
    {"Facility":"A3","FixedCost":300,"Capacity":20},
    {"Facility":"A4","FixedCost":375,"Capacity":30},
    {"Facility":"A5","FixedCost":500,"Capacity":40},
    {"Facility":"A6","FixedCost":200,"Capacity":20},
    {"Facility":"A7","FixedCost":260,"Capacity":25},
    {"Facility":"A8","FixedCost":220,"Capacity":30},
    {"Facility":"A9","FixedCost":320,"Capacity":35},
    {"Facility":"A10","FixedCost":280,"Capacity":20},
    {"Facility":"A11","FixedCost":350,"Capacity":40},
    {"Facility":"A12","FixedCost":420,"Capacity":25},
    {"Facility":"A13","FixedCost":470,"Capacity":30},
    {"Facility":"A14","FixedCost":520,"Capacity":50},
    {"Facility":"A15","FixedCost":560,"Capacity":45}
  ],
  "shipping_costs_matrix": [
    {"Origin":"A1","B1":8,"B2":4,"B3":3,"B4":6,"B5":7,"B6":5,"B7":9,"B8":8},
    {"Origin":"A2","B1":5,"B2":2,"B3":3,"B4":5,"B5":6,"B6":4,"B7":7,"B8":6},
    {"Origin":"A3","B1":4,"B2":3,"B3":4,"B4":6,"B5":5,"B6":5,"B7":6,"B8":7},
    {"Origin":"A4","B1":9,"B2":7,"B3":5,"B4":8,"B5":9,"B6":6,"B7":10,"B8":7},
    {"Origin":"A5","B1":10,"B2":4,"B3":2,"B4":6,"B5":8,"B6":5,"B7":7,"B8":3},
    {"Origin":"A6","B1":6,"B2":5,"B3":4,"B4":5,"B5":7,"B6":6,"B7":8,"B8":5},
    {"Origin":"A7","B1":7,"B2":6,"B3":5,"B4":4,"B5":6,"B6":7,"B7":9,"B8":6},
    {"Origin":"A8","B1":5,"B2":4,"B3":6,"B4":3,"B5":5,"B6":6,"B7":7,"B8":6},
    {"Origin":"A9","B1":8,"B2":7,"B3":6,"B4":7,"B5":9,"B6":8,"B7":10,"B8":7},
    {"Origin":"A10","B1":6,"B2":5,"B3":7,"B4":4,"B5":6,"B6":5,"B7":7,"B8":5},
    {"Origin":"A11","B1":9,"B2":6,"B3":4,"B4":6,"B5":8,"B6":7,"B7":9,"B8":6},
    {"Origin":"A12","B1":7,"B2":5,"B3":6,"B4":5,"B5":6,"B6":5,"B7":8,"B8":5},
    {"Origin":"A13","B1":8,"B2":6,"B3":5,"B4":6,"B5":7,"B6":6,"B7":8,"B8":7},
    {"Origin":"A14","B1":9,"B2":5,"B3":3,"B4":5,"B5":7,"B6":4,"B7":6,"B8":4},
    {"Origin":"A15","B1":10,"B2":6,"B3":4,"B4":5,"B5":8,"B6":5,"B7":7,"B8":5}
  ],
  "demand_vector": {
    "B1":30,
    "B2":25,
    "B3":20,
    "B4":35,
    "B5":25,
    "B6":30,
    "B7":25,
    "B8":30
  },
  "M":220
}