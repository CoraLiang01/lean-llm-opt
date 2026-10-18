##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of book $j$ placed on bookshelf $i$, for $i\in I$, $j\in J$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Shelf capacity: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\;\le\;C_i,\quad \forall i\in I$.

2. Domains: $x_{ij}\in \mathbb{Z}_{\ge 0}$.

$ I=\{1,2,\dots ,10\}$, $J=\{1,2,\dots ,25\}$.

##### Retrieved Information

**Shelves (capacity)**  
$C=\begin{bmatrix}
200\\
200\\
300\\
400\\
550\\
600\\
650\\
750\\
820\\
570
\end{bmatrix}$

**Books (value and weight)**  

$V=\begin{bmatrix}
50\\
70\\
30\\
60\\
80\\
90\\
40\\
100\\
55\\
75\\
65\\
95\\
45\\
85\\
70\\
110\\
50\\
60\\
120\\
100\\
52\\
68\\
38\\
58\\
82
\end{bmatrix}$  

$W=\begin{bmatrix}
10\\
20\\
5\\
15\\
25\\
30\\
12\\
35\\
10\\
20\\
18\\
28\\
8\\
22\\
25\\
40\\
14\\
16\\
50\\
30\\
11\\
19\\
7\\
14\\
24
\end{bmatrix}$  

**Product names (index $j$)**  
$1:\text{The Great Gatsby}$  
$2:\text{To Kill a Mockingbird}$  
$3:\text{1984}$  
$4:\text{Pride and Prejudice}$  
$5:\text{The Catcher in the Rye}$  
$6:\text{Moby Dick}$  
$7:\text{Jane Eyre}$  
$8:\text{War and Peace}$  
$9:\text{The Odyssey}$  
$10:\text{Crime and Punishment}$  
$11:\text{The Hobbit}$  
$12:\text{Brave New World}$  
$13:\text{Anna Karenina}$  
$14:\text{Wuthering Heights}$  
$15:\text{The Divine Comedy}$  
$16:\text{The Iliad}$  
$17:\text{Les Misérables}$  
$18:\text{Dracula}$  
$19:\text{Frankenstein}$  
$20:\text{The Brothers Karamazov}$  
$21:\text{Don Quixote}$  
$22:\text{One Hundred Years of Solitude}$  
$23:\text{Ulysses}$  
$24:\text{The Alchemist}$  
$25:\text{Meditations}$