Let $I$ be the set of products, indexed by $i$, with identifiers as in the "Product Name" column.

Let $x_i$ = number of units of product $i$ fulfilled (decision variable), for each product $i$.

Parameters (from the data):

- $r_i$ = Revenue for product $i$ (from "Revenue" column)
- $d_i$ = Demand for product $i$ (from "Demand" column)
- $s_i$ = Initial Inventory for product $i$ (from "Initial Inventory" column)

All variables $x_i$ are nonnegative integers.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to, for all $i \in I$:
\[
0 \leq x_i \leq \min\{d_i,\, s_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, using the retrieved data (in source order):

Let $I$ be the set of products:

\[
\begin{array}{llll}
\text{Product Name} & r_i & d_i & s_i \\
\hline
\text{Adams Group\_service} & 1003.56 & 10 & 80 \\
\text{Anderson-Leach\_against} & 549.92 & 6 & 40 \\
\text{Anderson-Valdez\_somebody} & 169.7 & 75 & 560 \\
\text{Anderson-White\_son} & 1285.81 & 121 & 820 \\
\text{Andrews LLC\_matter} & 1269.71 & 130 & 960 \\
\text{Andrews PLC\_enter} & 805.82 & 112 & 850 \\
\text{Andrews-Martin\_build} & 1134.19 & 125 & 950 \\
\text{Arias-Mendoza\_life} & 1479.23 & 54 & 360 \\
\text{Bennett and Sons\_down} & 139.14 & 75 & 580 \\
\text{Bennett, Foster and Moreno\_enter} & 518.67 & 102 & 800 \\
\text{Fernandez, Long and Nelson\_member} & 1088.57 & 48 & 340 \\
\text{Fernandez-Fischer\_million} & 371.56 & 4 & 30 \\
\text{Ferrell Inc\_prove} & 956.78 & 125 & 990 \\
\text{Fields, Christensen and Daniels\_nor} & 1283.7 & 9 & 70 \\
\text{Figueroa LLC\_involve} & 1184.54 & 132 & 890 \\
\text{Finley Group\_itself} & 377.8 & 29 & 240 \\
\text{Fisher Ltd\_speak} & 701.97 & 43 & 290 \\
\text{Fisher-Marshall\_cover} & 875.91 & 5 & 40 \\
\text{Fuller-Torres\_behavior} & 951.85 & 40 & 300 \\
\text{Fuller-Walters\_radio} & 869.95 & 48 & 350 \\
\text{Gallagher-Campbell\_dark} & 887.06 & 87 & 600 \\
\text{Gallagher-Kirby\_talk} & 1318.33 & 79 & 610 \\
\text{Gallagher-Parker\_recognize} & 448.61 & 20 & 160 \\
\text{Gonzalez PLC\_task} & 167.99 & 33 & 260 \\
\text{Gonzalez, Coleman and Le\_heavy} & 1068.1 & 60 & 480 \\
\text{Gonzalez, Lowe and Robinson\_education} & 872.16 & 25 & 200 \\
\text{Gonzalez-Horn\_light} & 483.26 & 42 & 290 \\
\text{Good, Davis and Smith\_station} & 1279.24 & 52 & 350 \\
\text{Goodman, Hughes and White\_realize} & 894.34 & 116 & 900 \\
\text{Goodwin PLC\_bad} & 367.91 & 95 & 760 \\
\text{Gordon PLC\_detail} & 178.22 & 66 & 440 \\
\text{Graham LLC\_stage} & 1030.8 & 25 & 170 \\
\text{Graham Ltd\_marriage} & 479.44 & 78 & 600 \\
\text{Graham-Swanson\_message} & 973.62 & 138 & 920 \\
\text{Grant, Cross and Bennett\_religious} & 1211.35 & 32 & 240 \\
\text{Grant, Mcdonald and Watson\_owner} & 467.93 & 28 & 190 \\
\text{Graves, Turner and Crawford\_wait} & 678.72 & 84 & 600 \\
\text{Gray, Smith and Barnes\_much} & 478.36 & 102 & 760 \\
\text{Green Inc\_direction} & 1196.95 & 112 & 850 \\
\text{Green-Rogers\_could} & 1118.63 & 7 & 50 \\
\text{Greene-Baxter\_them} & 972.1 & 62 & 470 \\
\text{Greer and Sons\_keep} & 338.6 & 109 & 790 \\
\text{Griffin, Boyle and Dawson\_anything} & 178.52 & 81 & 580 \\
\text{Harper Inc\_well} & 992.14 & 131 & 880 \\
\text{Harper Ltd\_off} & 498.77 & 26 & 170 \\
\text{Harrington, Sosa and Mccarty\_coach} & 965.34 & 109 & 790 \\
\text{Harris Group\_different} & 321.11 & 65 & 500 \\
\text{Harris and Sons\_audience} & 287.36 & 121 & 890 \\
\text{Harris and Sons\_fear} & 462.26 & 90 & 680 \\
\text{Harris, Hamilton and Rose\_contain} & 641.22 & 31 & 250 \\
\text{Harris, Stevens and Hall\_answer} & 1136.87 & 24 & 170 \\
\text{Harris-Bell\_painting} & 1385.88 & 122 & 830 \\
\text{Harris-Melton\_often} & 867.3 & 101 & 680 \\
\text{Harris-Rogers\_after} & 1175.92 & 76 & 590 \\
\text{Hart Group\_cold} & 153.61 & 100 & 700 \\
\text{Hutchinson, Roberts and Mcbride\_among} & 753.55 & 125 & 830 \\
\text{Ibarra, Jackson and Potter\_upon} & 245.81 & 45 & 360 \\
\text{Jackson Inc\_anyone} & 507.13 & 80 & 570 \\
\text{Jackson LLC\_report} & 513.64 & 64 & 510 \\
\text{Jackson, Collier and Barber\_result} & 1316.82 & 98 & 690 \\
\text{Jackson, White and Brown\_expect} & 1409.21 & 102 & 760 \\
\text{Jackson-Carroll\_six} & 1237.37 & 40 & 270 \\
\text{Lee-Jones\_soldier} & 1042.66 & 110 & 880 \\
\text{Lewis, Sanchez and Turner\_point} & 156.02 & 59 & 470 \\
\text{Lin LLC\_performance} & 728.62 & 97 & 670 \\
\text{Lindsey, Avila and Brown\_candidate} & 971.93 & 90 & 620 \\
\text{Lindsey-Strickland\_modern} & 1128.07 & 43 & 310 \\
\text{Little-Perez\_choice} & 256.42 & 96 & 670 \\
\text{Lloyd, Stone and Mcguire\_national} & 1064.72 & 21 & 150 \\
\text{Long, Hughes and Gallegos\_receive} & 348.43 & 55 & 440 \\
\text{Lopez PLC\_analysis} & 1183.94 & 30 & 210 \\
\end{array}
\]

The complete model is:

\[
\max \sum_{i \in I} r_i x_i
\]
subject to
\[
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

where all $r_i$, $d_i$, $s_i$ are as given above for each product $i$.