##### Sets

Let $I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12\}$ be the set of distribution centers (suppliers).

Let $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12\}$ be the set of customer groups.

##### Parameters

Customer group demands (from customer_demand.csv, in source order):

\[
\begin{aligned}
&d_{C1} = 52 \\
&d_{C2} = 80 \\
&d_{C3} = 392 \\
&d_{C4} = 103 \\
&d_{C5} = 32 \\
&d_{C6} = 1426 \\
&d_{C7} = 1024 \\
&d_{C8} = 2736 \\
&d_{C9} = 1129 \\
&d_{C10} = 676 \\
&d_{C11} = 2631 \\
&d_{C12} = 31 \\
\end{aligned}
\]

Distribution center supply capacities (from supply_capacity.csv, in source order):

\[
\begin{aligned}
&s_{S1} = 58 \\
&s_{S2} = 32 \\
&s_{S3} = 6161 \\
&s_{S4} = 4 \\
&s_{S5} = 47 \\
&s_{S6} = 178 \\
&s_{S7} = 142 \\
&s_{S8} = 164 \\
&s_{S9} = 1011 \\
&s_{S10} = 6 \\
&s_{S11} = 7081 \\
&s_{S12} = 948 \\
\end{aligned}
\]

Transportation costs $c_{ij}$ (from transportation_costs.csv, in source order):

- For each $i \in I$, $j \in J$, $c_{ij}$ is the value of "transportation_cost_to_$j$" for supplier $i$.

\[
\begin{aligned}
&\text{For } S1: \\
&\quad c_{S1,C1} = 134.72882437877243 \\
&\quad c_{S1,C2} = 72.30045141110347 \\
&\quad c_{S1,C3} = 37.9759927355347 \\
&\quad c_{S1,C4} = 611.84656497826 \\
&\quad c_{S1,C5} = 1650.3353902326157 \\
&\quad c_{S1,C6} = 32.97044486689555 \\
&\quad c_{S1,C7} = 34.99837373425947 \\
&\quad c_{S1,C8} = 73.25207370031997 \\
&\quad c_{S1,C9} = 165.752089835103 \\
&\quad c_{S1,C10} = 82.52035827662691 \\
&\quad c_{S1,C11} = 1538.830198350349 \\
&\quad c_{S1,C12} = 2320.1146477101956 \\

&\text{For } S2: \\
&\quad c_{S2,C1} = 23.835369186583733 \\
&\quad c_{S2,C2} = 1128.6332865368709 \\
&\quad c_{S2,C3} = 187.05459590556023 \\
&\quad c_{S2,C4} = 1.6077337129635412 \\
&\quad c_{S2,C5} = 2227.7036991312493 \\
&\quad c_{S2,C6} = 72.44693148708649 \\
&\quad c_{S2,C7} = 12.60074797346823 \\
&\quad c_{S2,C8} = 1078.7729548111274 \\
&\quad c_{S2,C9} = 383.7220569377865 \\
&\quad c_{S2,C10} = 82.8115250443967 \\
&\quad c_{S2,C11} = 1702.1724763823142 \\
&\quad c_{S2,C12} = 1925.8446970545492 \\

&\text{For } S3: \\
&\quad c_{S3,C1} = 1138.498759622659 \\
&\quad c_{S3,C2} = 231.8983422482863 \\
&\quad c_{S3,C3} = 44.1568226726957 \\
&\quad c_{S3,C4} = 962.5481640749796 \\
&\quad c_{S3,C5} = 980.1306032437444 \\
&\quad c_{S3,C6} = 1107.5157506568996 \\
&\quad c_{S3,C7} = 741.7073424442746 \\
&\quad c_{S3,C8} = 136.70204401709762 \\
&\quad c_{S3,C9} = 1182.5161115443857 \\
&\quad c_{S3,C10} = 664.1846792537777 \\
&\quad c_{S3,C11} = 36.70238978444623 \\
&\quad c_{S3,C12} = 1405.0506919253128 \\

&\text{For } S4: \\
&\quad c_{S4,C1} = 1043.7562803117191 \\
&\quad c_{S4,C2} = 24.207016063238527 \\
&\quad c_{S4,C3} = 1120.9890438994094 \\
&\quad c_{S4,C4} = 1027.5437638768522 \\
&\quad c_{S4,C5} = 893.4457577903337 \\
&\quad c_{S4,C6} = 1244.2626779205734 \\
&\quad c_{S4,C7} = 980.0297707948428 \\
&\quad c_{S4,C8} = 513.5650272201696 \\
&\quad c_{S4,C9} = 977.6321548115833 \\
&\quad c_{S4,C10} = 642.5520451224766 \\
&\quad c_{S4,C11} = 437.9556823253786 \\
&\quad c_{S4,C12} = 76.12498668603241 \\

&\text{For } S5: \\
&\quad c_{S5,C1} = 4.939930661779214 \\
&\quad c_{S5,C2} = 1278.1815735383598 \\
&\quad c_{S5,C3} = 549.7570743190075 \\
&\quad c_{S5,C4} = 21.335474111511868 \\
&\quad c_{S5,C5} = 98.32689279702352 \\
&\quad c_{S5,C6} = 452.0903462786745 \\
&\quad c_{S5,C7} = 595.5981933612871 \\
&\quad c_{S5,C8} = 70.84029746549007 \\
&\quad c_{S5,C9} = 0.001236242263106387 \\
&\quad c_{S5,C10} = 1783.0865677438942 \\
&\quad c_{S5,C11} = 1619.6154655942953 \\
&\quad c_{S5,C12} = 116.08497905117572 \\

&\text{For } S6: \\
&\quad c_{S6,C1} = 2105.5398596409113 \\
&\quad c_{S6,C2} = 1340.9770559580859 \\
&\quad c_{S6,C3} = 2077.2747284488696 \\
&\quad c_{S6,C4} = 2202.2515396165613 \\
&\quad c_{S6,C5} = 20.5336958780729 \\
&\quad c_{S6,C6} = 2514.0983598337716 \\
&\quad c_{S6,C7} = 2393.213466538681 \\
&\quad c_{S6,C8} = 1197.765543404598 \\
&\quad c_{S6,C9} = 102.32440248566687 \\
&\quad c_{S6,C10} = 788.4640306393768 \\
&\quad c_{S6,C11} = 818.9532260267922 \\
&\quad c_{S6,C12} = 17.249645794282493 \\

&\text{For } S7: \\
&\quad c_{S7,C1} = 61.36183063260923 \\
&\quad c_{S7,C2} = 113.23716964851326 \\
&\quad c_{S7,C3} = 50.231076475261865 \\
&\quad c_{S7,C4} = 1219.1951077346878 \\
&\quad c_{S7,C5} = 869.868986439361 \\
&\quad c_{S7,C6} = 58.3717723017115 \\
&\quad c_{S7,C7} = 957.3213131437595 \\
&\quad c_{S7,C8} = 168.9964798230943 \\
&\quad c_{S7,C9} = 1363.342713601386 \\
&\quad c_{S7,C10} = 519.3858364273143 \\
&\quad c_{S7,C11} = 483.3345987058458 \\
&\quad c_{S7,C12} = 1412.7652882009972 \\

&\text{For } S8: \\
&\quad c_{S8,C1} = 1169.3640988900754 \\
&\quad c_{S8,C2} = 1037.2170744709088 \\
&\quad c_{S8,C3} = 732.3577706907058 \\
&\quad c_{S8,C4} = 865.2255769415611 \\
&\quad c_{S8,C5} = 1510.0296187049466 \\
&\quad c_{S8,C6} = 780.853789775776 \\
&\quad c_{S8,C7} = 860.7971696230202 \\
&\quad c_{S8,C8} = 935.9126432821531 \\
&\quad c_{S8,C9} = 61.82504909829045 \\
&\quad c_{S8,C10} = 72.78392607232098 \\
&\quad c_{S8,C11} = 1479.1137195486635 \\
&\quad c_{S8,C12} = 65.87463475119084 \\

&\text{For } S9: \\
&\quad c_{S9,C1} = 7.936521667214095 \\
&\quad c_{S9,C2} = 1357.5387610434743 \\
&\quad c_{S9,C3} = 628.3001825422914 \\
&\quad c_{S9,C4} = 25.245141819018265 \\
&\quad c_{S9,C5} = 1760.0085249934855 \\
&\quad c_{S9,C6} = 604.6583535734275 \\
&\quad c_{S9,C7} = 696.677483277236 \\
&\quad c_{S9,C8} = 1586.4093183806565 \\
&\quad c_{S9,C9} = 89.68006942976479 \\
&\quad c_{S9,C10} = 1580.7262556456867 \\
&\quad c_{S9,C11} = 1423.5615412405277 \\
&\quad c_{S9,C12} = 2000.5637996766272 \\

&\text{For } S10: \\
&\quad c_{S10,C1} = 1685.3409758586952 \\
&\quad c_{S10,C2} = 437.39384912973503 \\
&\quad c_{S10,C3} = 1568.8939936485654 \\
&\quad c_{S10,C4} = 1486.9268967582923 \\
&\quad c_{S10,C5} = 498.37544466423503 \\
&\quad c_{S10,C6} = 1493.3860214089566 \\
&\quad c_{S10,C7} = 70.17952878640685 \\
&\quad c_{S10,C8} = 526.999992252258 \\
&\quad c_{S10,C9} = 1527.6606596831157 \\
&\quad c_{S10,C10} = 2.676413573774264 \\
&\quad c_{S10,C11} = 202.72671236450077 \\
&\quad c_{S10,C12} = 45.95308313010429 \\

&\text{For } S11: \\
&\quad c_{S11,C1} = 937.1094078301182 \\
&\quad c_{S11,C2} = 903.0141415848653 \\
&\quad c_{S11,C3} = 264.7208998642393 \\
&\quad c_{S11,C4} = 21.823340175910644 \\
&\quad c_{S11,C5} = 1661.105486348982 \\
&\quad c_{S11,C6} = 18.59349093795307 \\
&\quad c_{S11,C7} = 372.0817876391536 \\
&\quad c_{S11,C8} = 956.6955598743717 \\
&\quad c_{S11,C9} = 42.9120185256587 \\
&\quad c_{S11,C10} = 1274.3227526636908 \\
&\quad c_{S11,C11} = 1574.7961500081667 \\
&\quad c_{S11,C12} = 1826.428257280374 \\

&\text{For } S12: \\
&\quad c_{S12,C1} = 1685.7576850724427 \\
&\quad c_{S12,C2} = 377.3256672852414 \\
&\quad c_{S12,C3} = 1347.0162725120726 \\
&\quad c_{S12,C4} = 1737.020030508406 \\
&\quad c_{S12,C5} = 23.612177980608287 \\
&\quad c_{S12,C6} = 83.08521829288628 \\
&\quad c_{S12,C7} = 1476.1258241571823 \\
&\quad c_{S12,C8} = 530.0021927141197 \\
&\quad c_{S12,C9} = 1782.8442633270377 \\
&\quad c_{S12,C10} = 0.010486218015015225 \\
&\quad c_{S12,C11} = 11.1693204455916 \\
&\quad c_{S12,C12} = 963.9941814871104 \\
\end{aligned}
\]

##### Decision Variables

For all $i \in I$, $j \in J$:

$x_{ij} \geq 0$ : quantity shipped from distribution center $i$ to customer group $j$ (continuous).

##### Objective

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

##### Complete Model

\[
\begin{aligned}
&\min_{x_{ij} \geq 0} \quad \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
&\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
&\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
&x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
\end{aligned}
\]

where all sets, parameters, and coefficients are as listed above, in source order.