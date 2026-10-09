**Sets**

- Suppliers (stores): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$
- Customers: $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

**Parameters**

- Demand $d_j$ for each customer $j$:
  - $d_{\text{C1}} = 11$
  - $d_{\text{C2}} = 1148$
  - $d_{\text{C3}} = 54$
  - $d_{\text{C4}} = 833$
  - $d_{\text{C5}} = 154$
  - $d_{\text{C6}} = 551$
  - $d_{\text{C7}} = 7081$
  - $d_{\text{C8}} = 76$
  - $d_{\text{C9}} = 66$
  - $d_{\text{C10}} = 174$
  - $d_{\text{C11}} = 15$
  - $d_{\text{C12}} = 680$

- Supply capacity $s_i$ for each supplier $i$:
  - $s_{\text{S1}} = 4$
  - $s_{\text{S2}} = 575$
  - $s_{\text{S3}} = 1504$
  - $s_{\text{S4}} = 178$
  - $s_{\text{S5}} = 228$
  - $s_{\text{S6}} = 50$
  - $s_{\text{S7}} = 3$
  - $s_{\text{S8}} = 6148$
  - $s_{\text{S9}} = 6$
  - $s_{\text{S10}} = 10673$
  - $s_{\text{S11}} = 174$

- Transportation cost $c_{ij}$ from supplier $i$ to customer $j$:

|        |   C1   |    C2    |    C3    |    C4    |    C5    |    C6    |    C7    |    C8    |    C9    |   C10   |   C11   |   C12   |
|--------|--------|----------|----------|----------|----------|----------|----------|----------|----------|---------|---------|---------|
| S1     | 0.6391 | 49.7184  | 33.7586  | 1570.6731| 1370.4095| 57.3531  | 57.1830  | 54.9210  |1143.6809 |52.4913  |606.4434 |1192.4687|
| S2     |605.4786| 64.5356  |478.4779  | 887.0481 | 65.4611  | 71.9361  | 41.2902  | 70.3604  | 35.3589  |1472.7482| 0.6005  | 49.8685 |
| S3     |1139.0440| 4.7851  |1805.6214 |1302.8958 |2437.3212 |103.8037  |774.6558  | 4.5160   |879.7049  |162.7056 |1208.6135|110.1869 |
| S4     | 69.2699|2105.4854 | 869.6820 |1494.8986 | 310.5377 | 98.1546  |103.3692  |1758.8784 | 97.2854  | 94.6504 |1277.2515| 21.6362 |
| S5     |980.4114|899.3109  |1183.0326 | 402.0986 | 81.7886  |1115.6819 |123.8043  |1121.1469 | 0.0024   |1009.6452| 35.3480 |1625.4346|
| S6     |1246.7825|2105.7967|1014.3393 |1494.6681 | 362.0174 | 98.1714  |2170.4059 | 97.7319  | 97.2683  |1987.9908| 70.9440 |389.1598 |
| S7     | 57.1086| 23.8362  | 78.1057  | 742.8068 |1926.0797 |454.3790  |458.2901  |465.9308  | 28.1386  |524.6154 |997.5318 |104.4779 |
| S8     |981.2909|120.9013  |1625.8207 |1267.8229 |2569.6446 | 13.4718  |815.1525  |253.4235  | 43.7656  |275.9784 |1228.0699|103.4832 |
| S9     | 30.5328|1444.8595 | 173.5547 |1307.3913 | 965.2012 |1843.7769 |1483.6409 | 85.3221  |1353.5009 |1485.9154| 29.4238 | 26.6194 |
| S10    | 94.1109|1422.9971 |1470.7769 |1419.3382 | 38.9453  | 72.2011  |2040.4606 |1542.7026 |1803.8002 | 72.9437 |2181.4542|973.5516 |
| S11    |1032.9074|166.3018 |1620.4767 | 64.6683  |2000.5092 | 0.0029   | 47.0384  | 52.9922  |1115.6336 |129.7934 |1295.0978|2330.7682|

**Decision Variables**

- $x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$.

**Objective**

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

**Constraints**

1. **Demand satisfaction:** For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   That is,
   - $x_{\text{S1},j} + x_{\text{S2},j} + \cdots + x_{\text{S11},j} \geq d_j$ for each $j$.

2. **Supply capacity:** For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   That is,
   - $x_{i,\text{C1}} + x_{i,\text{C2}} + \cdots + x_{i,\text{C12}} \leq s_i$ for each $i$.

3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

**Full Numerical Model**

Minimize
\[
\begin{align*}
&0.639144476970582\,x_{\text{S1},\text{C1}} + 49.71842803015729\,x_{\text{S1},\text{C2}} + 33.75857739960576\,x_{\text{S1},\text{C3}} + 1570.673110465785\,x_{\text{S1},\text{C4}} \\
&+ 1370.4095474322417\,x_{\text{S1},\text{C5}} + 57.35307774277479\,x_{\text{S1},\text{C6}} + 57.18299486453194\,x_{\text{S1},\text{C7}} + 54.9209612366192\,x_{\text{S1},\text{C8}} \\
&+ 1143.680909226399\,x_{\text{S1},\text{C9}} + 52.49127007738756\,x_{\text{S1},\text{C10}} + 606.4434399601076\,x_{\text{S1},\text{C11}} + 1192.4686514332489\,x_{\text{S1},\text{C12}} \\
&+ 605.4786373569875\,x_{\text{S2},\text{C1}} + 64.53562572761275\,x_{\text{S2},\text{C2}} + 478.4779031378926\,x_{\text{S2},\text{C3}} + 887.0480739088434\,x_{\text{S2},\text{C4}} \\
&+ 65.46111249492031\,x_{\text{S2},\text{C5}} + 71.93605217833378\,x_{\text{S2},\text{C6}} + 41.29015388498019\,x_{\text{S2},\text{C7}} + 70.36038207491039\,x_{\text{S2},\text{C8}} \\
&+ 35.35892996332259\,x_{\text{S2},\text{C9}} + 1472.7481944140839\,x_{\text{S2},\text{C10}} + 0.6004591535232997\,x_{\text{S2},\text{C11}} + 49.86854015671519\,x_{\text{S2},\text{C12}} \\
&+ 1139.0440074582496\,x_{\text{S3},\text{C1}} + 4.785056325458736\,x_{\text{S3},\text{C2}} + 1805.6214229758102\,x_{\text{S3},\text{C3}} + 1302.8958147418275\,x_{\text{S3},\text{C4}} \\
&+ 2437.321229159901\,x_{\text{S3},\text{C5}} + 103.80368582531935\,x_{\text{S3},\text{C6}} + 774.6558236505713\,x_{\text{S3},\text{C7}} + 4.515988277174664\,x_{\text{S3},\text{C8}} \\
&+ 879.7048537066717\,x_{\text{S3},\text{C9}} + 162.70556734409897\,x_{\text{S3},\text{C10}} + 1208.613484750161\,x_{\text{S3},\text{C11}} + 110.18688517926226\,x_{\text{S3},\text{C12}} \\
&+ 69.26989890601938\,x_{\text{S4},\text{C1}} + 2105.485387219297\,x_{\text{S4},\text{C2}} + 869.6820232492624\,x_{\text{S4},\text{C3}} + 1494.8985656180187\,x_{\text{S4},\text{C4}} \\
&+ 310.5376623181487\,x_{\text{S4},\text{C5}} + 98.15455717980421\,x_{\text{S4},\text{C6}} + 103.36918486373995\,x_{\text{S4},\text{C7}} + 1758.8783768888936\,x_{\text{S4},\text{C8}} \\
&+ 97.28540713798621\,x_{\text{S4},\text{C9}} + 94.6504089308906\,x_{\text{S4},\text{C10}} + 1277.251451477508\,x_{\text{S4},\text{C11}} + 21.636190287574664\,x_{\text{S4},\text{C12}} \\
&+ 980.4114260089906\,x_{\text{S5},\text{C1}} + 899.3108831856057\,x_{\text{S5},\text{C2}} + 1183.032552702089\,x_{\text{S5},\text{C3}} + 402.0986161964097\,x_{\text{S5},\text{C4}} \\
&+ 81.78864123893297\,x_{\text{S5},\text{C5}} + 1115.6819455776936\,x_{\text{S5},\text{C6}} + 123.80427864011308\,x_{\text{S5},\text{C7}} + 1121.14687497079\,x_{\text{S5},\text{C8}} \\
&+ 0.0024451264081394235\,x_{\text{S5},\text{C9}} + 1009.6451734576028\,x_{\text{S5},\text{C10}} + 35.348018297991366\,x_{\text{S5},\text{C11}} + 1625.434626662855\,x_{\text{S5},\text{C12}} \\
&+ 1246.782499848912\,x_{\text{S6},\text{C1}} + 2105.7966672265507\,x_{\text{S6},\text{C2}} + 1014.3393213554372\,x_{\text{S6},\text{C3}} + 1494.6681439414933\,x_{\text{S6},\text{C4}} \\
&+ 362.0173906933171\,x_{\text{S6},\text{C5}} + 98.17142420905459\,x_{\text{S6},\text{C6}} + 2170.405867933639\,x_{\text{S6},\text{C7}} + 97.7318771093979\,x_{\text{S6},\text{C8}} \\
&+ 97.26834016110725\,x_{\text{S6},\text{C9}} + 1987.9907846635351\,x_{\text{S6},\text{C10}} + 70.94396914331519\,x_{\text{S6},\text{C11}} + 389.15980447937727\,x_{\text{S6},\text{C12}} \\
&+ 57.1086015288379\,x_{\text{S7},\text{C1}} + 23.836168859030245\,x_{\text{S7},\text{C2}} + 78.10572975165614\,x_{\text{S7},\text{C3}} + 742.8068113300407\,x_{\text{S7},\text{C4}} \\
&+ 1926.0796823736941\,x_{\text{S7},\text{C5}} + 454.3789956981779\,x_{\text{S7},\text{C6}} + 458.2901436941235\,x_{\text{S7},\text{C7}} + 465.9307664524444\,x_{\text{S7},\text{C8}} \\
&+ 28.138607069878855\,x_{\text{S7},\text{C9}} + 524.6154260270081\,x_{\text{S7},\text{C10}} + 997.531783848061\,x_{\text{S7},\text{C11}} + 104.47794493215576\,x_{\text{S7},\text{C12}} \\
&+ 981.2908605082814\,x_{\text{S8},\text{C1}} + 120.90130000942015\,x_{\text{S8},\text{C2}} + 1625.8206931791087\,x_{\text{S8},\text{C3}} + 1267.8229294135008\,x_{\text{S8},\text{C4}} \\
&+ 2569.6446053909003\,x_{\text{S8},\text{C5}} + 13.471811837256363\,x_{\text{S8},\text{C6}} + 815.1525428026629\,x_{\text{S8},\text{C7}} + 253.42349641458793\,x_{\text{S8},\text{C8}} \\
&+ 43.76562945456531\,x_{\text{S8},\text{C9}} + 275.9784134803488\,x_{\text{S8},\text{C10}} + 1228.06989342366\,x_{\text{S8},\text{C11}} + 103.48323020673556\,x_{\text{S8},\text{C12}} \\
&+ 30.532779511898102\,x_{\text{S9},\text{C1}} + 1444.8594995969975\,x_{\text{S9},\text{C2}} + 173.5547323639261\,x_{\text{S9},\text{C3}} + 1307.3913121142912\,x_{\text{S9},\text{C4}} \\
&+ 965.2012304156898\,x_{\text{S9},\text{C5}} + 1843.7769498110006\,x_{\text{S9},\text{C6}} + 1483.6408846054035\,x_{\text{S9},\text{C7}} + 85.32209952688736\,x_{\text{S9},\text{C8}} \\
&+ 1353.500934450796\,x_{\text{S9},\text{C9}} + 1485.9153764236357\,x_{\text{S9},\text{C10}} + 29.423790844675874\,x_{\text{S9},\text{C11}} + 26.619419605630917\,x_{\text{S9},\text{C12}} \\
&+ 94.11093956131819\,x_{\text{S10},\text{C1}} + 1422.9971302244805\,x_{\text{S10},\text{C2}} + 1470.776907673336\,x_{\text{S10},\text{C3}} + 1419.3382251704456\,x_{\text{S10},\text{C4}} \\
&+ 38.94527784177093\,x_{\text{S10},\text{C5}} + 72.20112949102915\,x_{\text{S10},\text{C6}} + 2040.4605902793303\,x_{\text{S10},\text{C7}} + 1542.702557551204\,x_{\text{S10},\text{C8}} \\
&+ 1803.8001691896025\,x_{\text{S10},\text{C9}} + 72.94365832842001\,x_{\text{S10},\text{C10}} + 2181.454205937846\,x_{\text{S10},\text{C11}} + 973.5515553238279\,x_{\text{S10},\text{C12}} \\
&+ 1032.9073835595004\,x_{\text{S11},\text{C1}} + 166.30184787458444\,x_{\text{S11},\text{C2}} + 1620.4767053028727\,x_{\text{S11},\text{C3}} + 64.668341345234\,x_{\text{S11},\text{C4}} \\
&+ 2000.50917314264\,x_{\text{S11},\text{C5}} + 0.0028957952749427158\,x_{\text{S11},\text{C6}} + 47.038371401381845\,x_{\text{S11},\text{C7}} + 52.99221132466169\,x_{\text{S11},\text{C8}} \\
&+ 1115.6336172632205\,x_{\text{S11},\text{C9}} + 129.79338189093912\,x_{\text{S11},\text{C10}} + 1295.09784482215\,x_{\text{S11},\text{C11}} + 2330.76820970791\,x_{\text{S11},\text{C12}}
\end{align*}
\]

Subject to:

For each customer $j$:
\[
\sum_{i \in I} x_{ij} \geq d_j
\]
That is,
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + \cdots + x_{\text{S11},\text{C1}} &\geq 11 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S11},\text{C2}} &\geq 1148 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + \cdots + x_{\text{S11},\text{C3}} &\geq 54 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + \cdots + x_{\text{S11},\text{C4}} &\geq 833 \\
x_{\text{S1},\text{C5}} + x_{\text{S2},\text{C5}} + \cdots + x_{\text{S11},\text{C5}} &\geq 154 \\
x_{\text{S1},\text{C6}} + x_{\text{S2},\text{C6}} + \cdots + x_{\text{S11},\text{C6}} &\geq 551 \\
x_{\text{S1},\text{C7}} + x_{\text{S2},\text{C7}} + \cdots + x_{\text{S11},\text{C7}} &\geq 7081 \\
x_{\text{S1},\text{C8}} + x_{\text{S2},\text{C8}} + \cdots + x_{\text{S11},\text{C8}} &\geq 76 \\
x_{\text{S1},\text{C9}} + x_{\text{S2},\text{C9}} + \cdots + x_{\text{S11},\text{C9}} &\geq 66 \\
x_{\text{S1},\text{C10}} + x_{\text{S2},\text{C10}} + \cdots + x_{\text{S11},\text{C10}} &\geq 174 \\
x_{\text{S1},\text{C11}} + x_{\text{S2},\text{C11}} + \cdots + x_{\text{S11},\text{C11}} &\geq 15 \\
x_{\text{S1},\text{C12}} + x_{\text{S2},\text{C12}} + \cdots + x_{\text{S11},\text{C12}} &\geq 680 \\
\end{align*}
\]

For each supplier $i$:
\[
\sum_{j \in J} x_{ij} \leq s_i
\]
That is,
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + \cdots + x_{\text{S1},\text{C12}} &\leq 4 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S2},\text{C12}} &\leq 575 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + \cdots + x_{\text{S3},\text{C12}} &\leq 1504 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + \cdots + x_{\text{S4},\text{C12}} &\leq 178 \\
x_{\text{S5},\text{C1}} + x_{\text{S5},\text{C2}} + \cdots + x_{\text{S5},\text{C12}} &\leq 228 \\
x_{\text{S6},\text{C1}} + x_{\text{S6},\text{C2}} + \cdots + x_{\text{S6},\text{C12}} &\leq 50 \\
x_{\text{S7},\text{C1}} + x_{\text{S7},\text{C2}} + \cdots + x_{\text{S7},\text{C12}} &\leq 3 \\
x_{\text{S8},\text{C1}} + x_{\text{S8},\text{C2}} + \cdots + x_{\text{S8},\text{C12}} &\leq 6148 \\
x_{\text{S9},\text{C1}} + x_{\text{S9},\text{C2}} + \cdots + x_{\text{S9},\text{C12}} &\leq 6 \\
x_{\text{S10},\text{C1}} + x_{\text{S10},\text{C2}} + \cdots + x_{\text{S10},\text{C12}} &\leq 10673 \\
x_{\text{S11},\text{C1}} + x_{\text{S11},\text{C2}} + \cdots + x_{\text{S11},\text{C12}} &\leq 174 \\
\end{align*}
\]

And for all $i \in I$, $j \in J$:
\[
x_{ij} \geq 0
\]