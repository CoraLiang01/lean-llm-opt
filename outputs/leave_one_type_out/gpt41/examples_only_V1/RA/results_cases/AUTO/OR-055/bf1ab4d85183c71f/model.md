Let:
- \( I \) be the set of display areas, indexed by \( i \), with DisplayID and Capacity from capacity.csv.
- \( J \) be the set of boat types, indexed by \( j \), with ProductName, Value, and Weight from products.csv.
- \( x_{ij} \) = number of units of boat type \( j \) placed in display area \( i \), integer, \( x_{ij} \geq 0 \).

Parameters:
From capacity.csv:
\[
\begin{array}{ll}
\text{DisplayID} & \text{Capacity} \\
1 & 356 \\
2 & 478 \\
3 & 305 \\
4 & 291 \\
5 & 168 \\
6 & 449 \\
7 & 139 \\
8 & 383 \\
9 & 472 \\
10 & 288 \\
11 & 320 \\
12 & 250 \\
13 & 402 \\
14 & 293 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
\text{Speedboat} & 69978 & 18 \\
\text{Fishing Boat} & 54011 & 42 \\
\text{Catamaran} & 36352 & 49 \\
\text{Yacht} & 51521 & 42 \\
\text{Sailboat} & 50415 & 41 \\
\text{Kayak} & 76109 & 48 \\
\text{Canoe} & 50462 & 22 \\
\text{Houseboat} & 28989 & 29 \\
\text{Pontoon} & 23318 & 45 \\
\text{Jet Ski} & 26142 & 14 \\
\text{Rowboat} & 42040 & 38 \\
\text{Hovercraft} & 85961 & 47 \\
\text{Cabin Cruiser} & 50142 & 45 \\
\text{Wakeboard Boat} & 48478 & 28 \\
\text{Dinghy} & 60953 & 24 \\
\text{Trawler} & 95265 & 39 \\
\text{Paddle Boat} & 22839 & 32 \\
\text{Submarine} & 90957 & 36 \\
\text{RIB} & 84652 & 14 \\
\text{Skiff} & 78991 & 16 \\
\end{array}
\]

Decision variables:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is the Value of ProductName \(j\) from products.csv.

Constraints:
For each display area \(i\) (DisplayID from capacity.csv):
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in I
\]
where \(\text{Weight}_j\) is the Weight of ProductName \(j\) from products.csv, and \(\text{Capacity}_i\) is the Capacity of DisplayID \(i\) from capacity.csv.

Variable domains:
\[
x_{ij} \in \{0, 1, 2, \ldots\} \qquad \forall i \in I, j \in J
\]

Summary:
- Decision variables: \(x_{ij}\) = number of units of boat type \(j\) in display area \(i\), integer, nonnegative.
- Objective: Maximize total value of all boats placed.
- Constraints: For each display area, total size (sum of weights of boats placed) does not exceed its capacity.
- All data and indices are explicitly aligned to the provided CSVs.