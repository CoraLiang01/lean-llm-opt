Let:
- i index the bookshelves, with BookshelfID from capacity.csv (i = 1,...,10)
- j index the books, with ProductName from products.csv (j = 1,...,25, in the order given)
- x_{ij} = number of units of book j placed on bookshelf i (integer, x_{ij} ≥ 0)

Parameters:
- C_i = Capacity of bookshelf i (from capacity.csv)
- v_j = Value of book j (from products.csv)
- w_j = Weight of book j (from products.csv)

Model:

Maximize total value:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j=1}^{25} v_j \cdot x_{ij}
\]
where v_j and the book order are:

1. The Great Gatsby: 50
2. To Kill a Mockingbird: 70
3. 1984: 30
4. Pride and Prejudice: 60
5. The Catcher in the Rye: 80
6. Moby Dick: 90
7. Jane Eyre: 40
8. War and Peace: 100
9. The Odyssey: 55
10. Crime and Punishment: 75
11. The Hobbit: 65
12. Brave New World: 95
13. Anna Karenina: 45
14. Wuthering Heights: 85
15. The Divine Comedy: 70
16. The Iliad: 110
17. Les Misérables: 50
18. Dracula: 60
19. Frankenstein: 120
20. The Brothers Karamazov: 100
21. Don Quixote: 52
22. One Hundred Years of Solitude: 68
23. Ulysses: 38
24. The Alchemist: 58
25. Meditations: 82

Subject to bookshelf capacity constraints (for each i = 1,...,10):
\[
\sum_{j=1}^{25} w_j \cdot x_{ij} \leq C_i
\]
where w_j and the book order are:

1. The Great Gatsby: 10
2. To Kill a Mockingbird: 20
3. 1984: 5
4. Pride and Prejudice: 15
5. The Catcher in the Rye: 25
6. Moby Dick: 30
7. Jane Eyre: 12
8. War and Peace: 35
9. The Odyssey: 10
10. Crime and Punishment: 20
11. The Hobbit: 18
12. Brave New World: 28
13. Anna Karenina: 8
14. Wuthering Heights: 22
15. The Divine Comedy: 25
16. The Iliad: 40
17. Les Misérables: 14
18. Dracula: 16
19. Frankenstein: 50
20. The Brothers Karamazov: 30
21. Don Quixote: 11
22. One Hundred Years of Solitude: 19
23. Ulysses: 7
24. The Alchemist: 14
25. Meditations: 24

Bookshelf capacities (C_i) are:

BookshelfID | Capacity
--- | ---
1 | 200
2 | 200
3 | 300
4 | 400
5 | 550
6 | 600
7 | 650
8 | 750
9 | 820
10 | 570

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,25\}
\]

Full numerical formulation:

Maximize:
\[
\sum_{i=1}^{10} \Big(
50 x_{i1} + 70 x_{i2} + 30 x_{i3} + 60 x_{i4} + 80 x_{i5} + 90 x_{i6} + 40 x_{i7} + 100 x_{i8} + 55 x_{i9} + 75 x_{i10} + 65 x_{i11} + 95 x_{i12} + 45 x_{i13} + 85 x_{i14} + 70 x_{i15} + 110 x_{i16} + 50 x_{i17} + 60 x_{i18} + 120 x_{i19} + 100 x_{i20} + 52 x_{i21} + 68 x_{i22} + 38 x_{i23} + 58 x_{i24} + 82 x_{i25}
\Big)
\]

Subject to, for each bookshelf i (BookshelfID as below):

For i=1 (BookshelfID=1, Capacity=200):
\[
10 x_{1,1} + 20 x_{1,2} + 5 x_{1,3} + 15 x_{1,4} + 25 x_{1,5} + 30 x_{1,6} + 12 x_{1,7} + 35 x_{1,8} + 10 x_{1,9} + 20 x_{1,10} + 18 x_{1,11} + 28 x_{1,12} + 8 x_{1,13} + 22 x_{1,14} + 25 x_{1,15} + 40 x_{1,16} + 14 x_{1,17} + 16 x_{1,18} + 50 x_{1,19} + 30 x_{1,20} + 11 x_{1,21} + 19 x_{1,22} + 7 x_{1,23} + 14 x_{1,24} + 24 x_{1,25} \leq 200
\]

For i=2 (BookshelfID=2, Capacity=200):
(same as above, with x_{2,j})

For i=3 (BookshelfID=3, Capacity=300):
(same as above, with x_{3,j}, ≤ 300)

For i=4 (BookshelfID=4, Capacity=400):
(same as above, with x_{4,j}, ≤ 400)

For i=5 (BookshelfID=5, Capacity=550):
(same as above, with x_{5,j}, ≤ 550)

For i=6 (BookshelfID=6, Capacity=600):
(same as above, with x_{6,j}, ≤ 600)

For i=7 (BookshelfID=7, Capacity=650):
(same as above, with x_{7,j}, ≤ 650)

For i=8 (BookshelfID=8, Capacity=750):
(same as above, with x_{8,j}, ≤ 750)

For i=9 (BookshelfID=9, Capacity=820):
(same as above, with x_{9,j}, ≤ 820)

For i=10 (BookshelfID=10, Capacity=570):
(same as above, with x_{10,j}, ≤ 570)

Variable domains:
\[
x_{ij} \in \{0,1,2,\ldots\} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,25\}
\]

All coefficients and identifiers are as given in the supplied files.