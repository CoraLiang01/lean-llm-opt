##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center $i$ to customer group $j$, for all $i \in I$ and $j \in J$ (continuous).

##### Sets and Indices

- $I = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$ (distribution centers, in source order)
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$ (customer groups, in source order)

##### Parameters

- Demands $d_j$ (from customer_demand.csv):

  - $d_{\text{demand1}} = 9$
  - $d_{\text{demand2}} = 66$
  - $d_{\text{demand3}} = 56$
  - $d_{\text{demand4}} = 17$
  - $d_{\text{demand5}} = 43$
  - $d_{\text{demand6}} = 62$
  - $d_{\text{demand7}} = 10$
  - $d_{\text{demand8}} = 37$

- Supply capacities $s_i$ (from supply_capacity.csv):

  - $s_{\text{supplier1}} = 60$
  - $s_{\text{supplier2}} = 22$
  - $s_{\text{supplier3}} = 16$
  - $s_{\text{supplier4}} = 14$
  - $s_{\text{supplier5}} = 19$
  - $s_{\text{supplier6}} = 70$
  - $s_{\text{supplier7}} = 60$
  - $s_{\text{supplier8}} = 39$

- Transportation costs $c_{ij}$ (from transportation_costs.csv, in source order):

  - For $i = \text{supply1}$:
    - $c_{\text{supply1},\text{demand1}} = 12$
    - $c_{\text{supply1},\text{demand2}} = 0.03020736643461065$
    - $c_{\text{supply1},\text{demand3}} = 229.50723504640203$
    - $c_{\text{supply1},\text{demand4}} = 198.62356558205792$
    - $c_{\text{supply1},\text{demand5}} = 12.995050640153751$
    - $c_{\text{supply1},\text{demand6}} = 211.20732124396406$
    - $c_{\text{supply1},\text{demand7}} = 134.9442985029274$
    - $c_{\text{supply1},\text{demand8}} = 9.822206398831067$
    - $c_{\text{supply1},\text{demand8}} = 11.394077543225675$

  - For $i = \text{supply2}$:
    - $c_{\text{supply2},\text{demand1}} = 232.34691308087835$
    - $c_{\text{supply2},\text{demand2}} = 3.6258726438627473$
    - $c_{\text{supply2},\text{demand3}} = 0.28605434149404785$
    - $c_{\text{supply2},\text{demand4}} = 45.73127693242935$
    - $c_{\text{supply2},\text{demand5}} = 2.8304796563034573$
    - $c_{\text{supply2},\text{demand6}} = 107.05891033185472$
    - $c_{\text{supply2},\text{demand7}} = 299.96317913389305$
    - $c_{\text{supply2},\text{demand8}} = 23.79935436307657$

  - For $i = \text{supply3}$:
    - $c_{\text{supply3},\text{demand1}} = 11.061938334356302$
    - $c_{\text{supply3},\text{demand2}} = 0.2041995326579051$
    - $c_{\text{supply3},\text{demand3}} = 0.2789447278030927$
    - $c_{\text{supply3},\text{demand4}} = 45.721912724349636$
    - $c_{\text{supply3},\text{demand5}} = 59.54895565737313$
    - $c_{\text{supply3},\text{demand6}} = 5.097536739581239$
    - $c_{\text{supply3},\text{demand7}} = 300.00118415135785$
    - $c_{\text{supply3},\text{demand8}} = 23.711282707746893$

  - For $i = \text{supply4}$:
    - $c_{\text{supply4},\text{demand1}} = 235.1794835706472$
    - $c_{\text{supply4},\text{demand2}} = 43.794668963036194$
    - $c_{\text{supply4},\text{demand3}} = 40.709846782945924$
    - $c_{\text{supply4},\text{demand4}} = 0.07774496620087613$
    - $c_{\text{supply4},\text{demand5}} = 4.237728183419554$
    - $c_{\text{supply4},\text{demand6}} = 131.70915517494691$
    - $c_{\text{supply4},\text{demand7}} = 296.55587567706743$
    - $c_{\text{supply4},\text{demand8}} = 29.810940017561297$

  - For $i = \text{supply5}$:
    - $c_{\text{supply5},\text{demand1}} = 211.85808746383796$
    - $c_{\text{supply5},\text{demand2}} = 47.60180876530328$
    - $c_{\text{supply5},\text{demand3}} = 50.04007716193931$
    - $c_{\text{supply5},\text{demand4}} = 86.14548807358399$
    - $c_{\text{supply5},\text{demand5}} = 0.06197897916874956$
    - $c_{\text{supply5},\text{demand6}} = 5.3345515296262205$
    - $c_{\text{supply5},\text{demand7}} = 270.06290423798396$
    - $c_{\text{supply5},\text{demand8}} = 3.853933133973331$

  - For $i = \text{supply6}$:
    - $c_{\text{supply6},\text{demand1}} = 6.45506633554524$
    - $c_{\text{supply6},\text{demand2}} = 88.16323623354015$
    - $c_{\text{supply6},\text{demand3}} = 5.047091671641611$
    - $c_{\text{supply6},\text{demand4}} = 151.46120287365497$
    - $c_{\text{supply6},\text{demand5}} = 5.290760161059401$
    - $c_{\text{supply6},\text{demand6}} = 0.04602205335871525$
    - $c_{\text{supply6},\text{demand7}} = 9.93670660180487$
    - $c_{\text{supply6},\text{demand8}} = 103.75460989446313$

  - For $i = \text{supply7}$:
    - $c_{\text{supply7},\text{demand1}} = 174.27229047340035$
    - $c_{\text{supply7},\text{demand2}} = 250.58223528739327$
    - $c_{\text{supply7},\text{demand3}} = 253.90413041857263$
    - $c_{\text{supply7},\text{demand4}} = 16.235467318386764$
    - $c_{\text{supply7},\text{demand5}} = 12.643140514778086$
    - $c_{\text{supply7},\text{demand6}} = 175.0672824108511$
    - $c_{\text{supply7},\text{demand7}} = 2.983839625303656$
    - $c_{\text{supply7},\text{demand8}} = 317.0655193866389$

  - For $i = \text{supply8}$:
    - $c_{\text{supply8},\text{demand1}} = 207.87006253790491$
    - $c_{\text{supply8},\text{demand2}} = 1.517168471518212$
    - $c_{\text{supply8},\text{demand3}} = 24.027239288137153$
    - $c_{\text{supply8},\text{demand4}} = 27.133999276450346$
    - $c_{\text{supply8},\text{demand5}} = 73.20672468851855$
    - $c_{\text{supply8},\text{demand6}} = 125.72910359893308$
    - $c_{\text{supply8},\text{demand7}} = 15.463103251642147$
    - $c_{\text{supply8},\text{demand8}} = 0.20164987511903337$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each customer group must receive at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity** (each distribution center cannot ship more than its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Numerical Model

\[
\begin{align*}
\min\quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{i,\text{demand1}} \geq 9 \\
& \sum_{i \in I} x_{i,\text{demand2}} \geq 66 \\
& \sum_{i \in I} x_{i,\text{demand3}} \geq 56 \\
& \sum_{i \in I} x_{i,\text{demand4}} \geq 17 \\
& \sum_{i \in I} x_{i,\text{demand5}} \geq 43 \\
& \sum_{i \in I} x_{i,\text{demand6}} \geq 62 \\
& \sum_{i \in I} x_{i,\text{demand7}} \geq 10 \\
& \sum_{i \in I} x_{i,\text{demand8}} \geq 37 \\
& \sum_{j \in J} x_{\text{supply1},j} \leq 60 \\
& \sum_{j \in J} x_{\text{supply2},j} \leq 22 \\
& \sum_{j \in J} x_{\text{supply3},j} \leq 16 \\
& \sum_{j \in J} x_{\text{supply4},j} \leq 14 \\
& \sum_{j \in J} x_{\text{supply5},j} \leq 19 \\
& \sum_{j \in J} x_{\text{supply6},j} \leq 70 \\
& \sum_{j \in J} x_{\text{supply7},j} \leq 60 \\
& \sum_{j \in J} x_{\text{supply8},j} \leq 39 \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
\]

where $c_{ij}$ are as listed above, and all indices and coefficients are preserved in source order.