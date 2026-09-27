Let $x_{ij}$ be the number of units of book $j$ (ProductName) to be placed on bookshelf $i$ (BookshelfID). All $x_{ij}$ are nonnegative integers.

Parameters:

- Let $B$ be the set of bookshelves: $B = \{1,2,3,4,5,6,7,8,9,10\}$
- Let $P$ be the set of books (ProductName), as listed below.
- $c_i$ = Capacity of bookshelf $i$
- $v_j$ = Value of book $j$
- $w_j$ = Weight of book $j$

Book data (ProductName, Value $v_j$, Weight $w_j$):

- The Great Gatsby: $v_j=50$, $w_j=10$
- To Kill a Mockingbird: $v_j=70$, $w_j=20$
- 1984: $v_j=30$, $w_j=5$
- Pride and Prejudice: $v_j=60$, $w_j=15$
- The Catcher in the Rye: $v_j=80$, $w_j=25$
- Moby Dick: $v_j=90$, $w_j=30$
- Jane Eyre: $v_j=40$, $w_j=12$
- War and Peace: $v_j=100$, $w_j=35$
- The Odyssey: $v_j=55$, $w_j=10$
- Crime and Punishment: $v_j=75$, $w_j=20$
- The Hobbit: $v_j=65$, $w_j=18$
- Brave New World: $v_j=95$, $w_j=28$
- Anna Karenina: $v_j=45$, $w_j=8$
- Wuthering Heights: $v_j=85$, $w_j=22$
- The Divine Comedy: $v_j=70$, $w_j=25$
- The Iliad: $v_j=110$, $w_j=40$
- Les Misérables: $v_j=50$, $w_j=14$
- Dracula: $v_j=60$, $w_j=16$
- Frankenstein: $v_j=120$, $w_j=50$
- The Brothers Karamazov: $v_j=100$, $w_j=30$
- Don Quixote: $v_j=52$, $w_j=11$
- One Hundred Years of Solitude: $v_j=68$, $w_j=19$
- Ulysses: $v_j=38$, $w_j=7$
- The Alchemist: $v_j=58$, $w_j=14$
- Meditations: $v_j=82$, $w_j=24$

Bookshelf data (BookshelfID $i$, Capacity $c_i$):

- 1: $c_1=200$
- 2: $c_2=200$
- 3: $c_3=300$
- 4: $c_4=400$
- 5: $c_5=550$
- 6: $c_6=600$
- 7: $c_7=650$
- 8: $c_8=750$
- 9: $c_9=820$
- 10: $c_{10}=570$

Mathematical Model:

Objective:
\[
\max \sum_{i \in B} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to (for all $i \in B$):

\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in B,\, j \in P
\]

Where:
- $x_{ij}$: number of units of book $j$ placed on bookshelf $i$ (integer, $\geq 0$)
- $v_j$: value of book $j$ (see above)
- $w_j$: weight of book $j$ (see above)
- $c_i$: capacity of bookshelf $i$ (see above)