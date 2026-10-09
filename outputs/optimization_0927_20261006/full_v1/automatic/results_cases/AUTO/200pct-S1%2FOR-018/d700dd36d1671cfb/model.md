##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier (distribution center) $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- Demand for each customer group $j$ ($d_j$):

  - $d_{\text{C1}} = 52$
  - $d_{\text{C2}} = 80$
  - $d_{\text{C3}} = 392$
  - $d_{\text{C4}} = 103$
  - $d_{\text{C5}} = 32$
  - $d_{\text{C6}} = 1426$
  - $d_{\text{C7}} = 1024$
  - $d_{\text{C8}} = 2736$
  - $d_{\text{C9}} = 1129$
  - $d_{\text{C10}} = 676$
  - $d_{\text{C11}} = 2631$
  - $d_{\text{C12}} = 31$

- Supply capacity for each supplier $i$ ($s_i$):

  - $s_{\text{S1}} = 58$
  - $s_{\text{S2}} = 32$
  - $s_{\text{S3}} = 6161$
  - $s_{\text{S4}} = 4$
  - $s_{\text{S5}} = 47$
  - $s_{\text{S6}} = 178$
  - $s_{\text{S7}} = 142$
  - $s_{\text{S8}} = 164$
  - $s_{\text{S9}} = 1011$
  - $s_{\text{S10}} = 6$
  - $s_{\text{S11}} = 7081$
  - $s_{\text{S12}} = 948$

- Transportation cost per unit from supplier $i$ to customer $j$ ($c_{ij}$):

  - For S1: 
    - $c_{\text{S1},\text{C1}} = 134.72882437877243$
    - $c_{\text{S1},\text{C2}} = 72.30045141110347$
    - $c_{\text{S1},\text{C3}} = 37.9759927355347$
    - $c_{\text{S1},\text{C4}} = 611.84656497826$
    - $c_{\text{S1},\text{C5}} = 1650.3353902326157$
    - $c_{\text{S1},\text{C6}} = 32.97044486689555$
    - $c_{\text{S1},\text{C7}} = 34.99837373425947$
    - $c_{\text{S1},\text{C8}} = 73.25207370031997$
    - $c_{\text{S1},\text{C9}} = 165.752089835103$
    - $c_{\text{S1},\text{C10}} = 82.52035827662691$
    - $c_{\text{S1},\text{C11}} = 1538.830198350349$
    - $c_{\text{S1},\text{C12}} = 2320.1146477101956$
  - For S2:
    - $c_{\text{S2},\text{C1}} = 23.835369186583733$
    - $c_{\text{S2},\text{C2}} = 1128.6332865368709$
    - $c_{\text{S2},\text{C3}} = 187.05459590556023$
    - $c_{\text{S2},\text{C4}} = 1.6077337129635412$
    - $c_{\text{S2},\text{C5}} = 2227.7036991312493$
    - $c_{\text{S2},\text{C6}} = 72.44693148708649$
    - $c_{\text{S2},\text{C7}} = 12.60074797346823$
    - $c_{\text{S2},\text{C8}} = 1078.7729548111274$
    - $c_{\text{S2},\text{C9}} = 383.7220569377865$
    - $c_{\text{S2},\text{C10}} = 82.8115250443967$
    - $c_{\text{S2},\text{C11}} = 1702.1724763823142$
    - $c_{\text{S2},\text{C12}} = 1925.8446970545492$
  - For S3:
    - $c_{\text{S3},\text{C1}} = 1138.498759622659$
    - $c_{\text{S3},\text{C2}} = 231.8983422482863$
    - $c_{\text{S3},\text{C3}} = 44.1568226726957$
    - $c_{\text{S3},\text{C4}} = 962.5481640749796$
    - $c_{\text{S3},\text{C5}} = 980.1306032437444$
    - $c_{\text{S3},\text{C6}} = 1107.5157506568996$
    - $c_{\text{S3},\text{C7}} = 741.7073424442746$
    - $c_{\text{S3},\text{C8}} = 136.70204401709762$
    - $c_{\text{S3},\text{C9}} = 1182.5161115443857$
    - $c_{\text{S3},\text{C10}} = 664.1846792537777$
    - $c_{\text{S3},\text{C11}} = 36.70238978444623$
    - $c_{\text{S3},\text{C12}} = 1405.0506919253128$
  - For S4:
    - $c_{\text{S4},\text{C1}} = 1043.7562803117191$
    - $c_{\text{S4},\text{C2}} = 24.207016063238527$
    - $c_{\text{S4},\text{C3}} = 1120.9890438994094$
    - $c_{\text{S4},\text{C4}} = 1027.5437638768522$
    - $c_{\text{S4},\text{C5}} = 893.4457577903337$
    - $c_{\text{S4},\text{C6}} = 1244.2626779205734$
    - $c_{\text{S4},\text{C7}} = 980.0297707948428$
    - $c_{\text{S4},\text{C8}} = 513.5650272201696$
    - $c_{\text{S4},\text{C9}} = 977.6321548115833$
    - $c_{\text{S4},\text{C10}} = 642.5520451224766$
    - $c_{\text{S4},\text{C11}} = 437.9556823253786$
    - $c_{\text{S4},\text{C12}} = 76.12498668603241$
  - For S5:
    - $c_{\text{S5},\text{C1}} = 4.939930661779214$
    - $c_{\text{S5},\text{C2}} = 1278.1815735383598$
    - $c_{\text{S5},\text{C3}} = 549.7570743190075$
    - $c_{\text{S5},\text{C4}} = 21.335474111511868$
    - $c_{\text{S5},\text{C5}} = 98.32689279702352$
    - $c_{\text{S5},\text{C6}} = 452.0903462786745$
    - $c_{\text{S5},\text{C7}} = 595.5981933612871$
    - $c_{\text{S5},\text{C8}} = 70.84029746549007$
    - $c_{\text{S5},\text{C9}} = 0.001236242263106387$
    - $c_{\text{S5},\text{C10}} = 1783.0865677438942$
    - $c_{\text{S5},\text{C11}} = 1619.6154655942953$
    - $c_{\text{S5},\text{C12}} = 116.08497905117572$
  - For S6:
    - $c_{\text{S6},\text{C1}} = 2105.5398596409113$
    - $c_{\text{S6},\text{C2}} = 1340.9770559580859$
    - $c_{\text{S6},\text{C3}} = 2077.2747284488696$
    - $c_{\text{S6},\text{C4}} = 2202.2515396165613$
    - $c_{\text{S6},\text{C5}} = 20.5336958780729$
    - $c_{\text{S6},\text{C6}} = 2514.0983598337716$
    - $c_{\text{S6},\text{C7}} = 2393.213466538681$
    - $c_{\text{S6},\text{C8}} = 1197.765543404598$
    - $c_{\text{S6},\text{C9}} = 102.32440248566687$
    - $c_{\text{S6},\text{C10}} = 788.4640306393768$
    - $c_{\text{S6},\text{C11}} = 818.9532260267922$
    - $c_{\text{S6},\text{C12}} = 17.249645794282493$
  - For S7:
    - $c_{\text{S7},\text{C1}} = 61.36183063260923$
    - $c_{\text{S7},\text{C2}} = 113.23716964851326$
    - $c_{\text{S7},\text{C3}} = 50.231076475261865$
    - $c_{\text{S7},\text{C4}} = 1219.1951077346878$
    - $c_{\text{S7},\text{C5}} = 869.868986439361$
    - $c_{\text{S7},\text{C6}} = 58.3717723017115$
    - $c_{\text{S7},\text{C7}} = 957.3213131437595$
    - $c_{\text{S7},\text{C8}} = 168.9964798230943$
    - $c_{\text{S7},\text{C9}} = 1363.342713601386$
    - $c_{\text{S7},\text{C10}} = 519.3858364273143$
    - $c_{\text{S7},\text{C11}} = 483.3345987058458$
    - $c_{\text{S7},\text{C12}} = 1412.7652882009972$
  - For S8:
    - $c_{\text{S8},\text{C1}} = 1169.3640988900754$
    - $c_{\text{S8},\text{C2}} = 1037.2170744709088$
    - $c_{\text{S8},\text{C3}} = 732.3577706907058$
    - $c_{\text{S8},\text{C4}} = 865.2255769415611$
    - $c_{\text{S8},\text{C5}} = 1510.0296187049466$
    - $c_{\text{S8},\text{C6}} = 780.853789775776$
    - $c_{\text{S8},\text{C7}} = 860.7971696230202$
    - $c_{\text{S8},\text{C8}} = 935.9126432821531$
    - $c_{\text{S8},\text{C9}} = 61.82504909829045$
    - $c_{\text{S8},\text{C10}} = 72.78392607232098$
    - $c_{\text{S8},\text{C11}} = 1479.1137195486635$
    - $c_{\text{S8},\text{C12}} = 65.87463475119084$
  - For S9:
    - $c_{\text{S9},\text{C1}} = 7.936521667214095$
    - $c_{\text{S9},\text{C2}} = 1357.5387610434743$
    - $c_{\text{S9},\text{C3}} = 628.3001825422914$
    - $c_{\text{S9},\text{C4}} = 25.245141819018265$
    - $c_{\text{S9},\text{C5}} = 1760.0085249934855$
    - $c_{\text{S9},\text{C6}} = 604.6583535734275$
    - $c_{\text{S9},\text{C7}} = 696.677483277236$
    - $c_{\text{S9},\text{C8}} = 1586.4093183806565$
    - $c_{\text{S9},\text{C9}} = 89.68006942976479$
    - $c_{\text{S9},\text{C10}} = 1580.7262556456867$
    - $c_{\text{S9},\text{C11}} = 1423.5615412405277$
    - $c_{\text{S9},\text{C12}} = 2000.5637996766272$
  - For S10:
    - $c_{\text{S10},\text{C1}} = 1685.3409758586952$
    - $c_{\text{S10},\text{C2}} = 437.39384912973503$
    - $c_{\text{S10},\text{C3}} = 1568.8939936485654$
    - $c_{\text{S10},\text{C4}} = 1486.9268967582923$
    - $c_{\text{S10},\text{C5}} = 498.37544466423503$
    - $c_{\text{S10},\text{C6}} = 1493.3860214089566$
    - $c_{\text{S10},\text{C7}} = 70.17952878640685$
    - $c_{\text{S10},\text{C8}} = 526.999992252258$
    - $c_{\text{S10},\text{C9}} = 1527.6606596831157$
    - $c_{\text{S10},\text{C10}} = 2.676413573774264$
    - $c_{\text{S10},\text{C11}} = 202.72671236450077$
    - $c_{\text{S10},\text{C12}} = 45.95308313010429$
  - For S11:
    - $c_{\text{S11},\text{C1}} = 937.1094078301182$
    - $c_{\text{S11},\text{C2}} = 903.0141415848653$
    - $c_{\text{S11},\text{C3}} = 264.7208998642393$
    - $c_{\text{S11},\text{C4}} = 21.823340175910644$
    - $c_{\text{S11},\text{C5}} = 1661.105486348982$
    - $c_{\text{S11},\text{C6}} = 18.59349093795307$
    - $c_{\text{S11},\text{C7}} = 372.0817876391536$
    - $c_{\text{S11},\text{C8}} = 956.6955598743717$
    - $c_{\text{S11},\text{C9}} = 42.9120185256587$
    - $c_{\text{S11},\text{C10}} = 1274.3227526636908$
    - $c_{\text{S11},\text{C11}} = 1574.7961500081667$
    - $c_{\text{S11},\text{C12}} = 1826.428257280374$
  - For S12:
    - $c_{\text{S12},\text{C1}} = 1685.7576850724427$
    - $c_{\text{S12},\text{C2}} = 377.3256672852414$
    - $c_{\text{S12},\text{C3}} = 1347.0162725120726$
    - $c_{\text{S12},\text{C4}} = 1737.020030508406$
    - $c_{\text{S12},\text{C5}} = 23.612177980608287$
    - $c_{\text{S12},\text{C6}} = 83.08521829288628$
    - $c_{\text{S12},\text{C7}} = 1476.1258241571823$
    - $c_{\text{S12},\text{C8}} = 530.0021927141197$
    - $c_{\text{S12},\text{C9}} = 1782.8442633270377$
    - $c_{\text{S12},\text{C10}} = 0.010486218015015225$
    - $c_{\text{S12},\text{C11}} = 11.1693204455916$
    - $c_{\text{S12},\text{C12}} = 963.9941814871104$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** For each customer group $j$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity:** For each supplier $i$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Complete Model

\[
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
\]

where all $d_j$, $s_i$, and $c_{ij}$ are as listed above.