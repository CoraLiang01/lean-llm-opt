##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of book $j$ placed on bookshelf $i$.

##### Objective Function

$\displaystyle \max \sum_{i}\sum_{j}v_{j}\,x_{ij}$

##### Constraints

1. Shelf capacity: $\displaystyle \sum_{j}w_{j}\,x_{ij}\leq C_{i},\quad\forall i$  
2. Non‑negativity and integrality: $x_{ij}\in \mathbb{Z}_{\ge 0},\quad\forall i,j$

###### Retrieved Information

{
  "shelves": [
    {"BookshelfID":"1","Capacity":"200"},
    {"BookshelfID":"2","Capacity":"200"},
    {"BookshelfID":"3","Capacity":"300"},
    {"BookshelfID":"4","Capacity":"400"},
    {"BookshelfID":"5","Capacity":"550"},
    {"BookshelfID":"6","Capacity":"600"},
    {"BookshelfID":"7","Capacity":"650"},
    {"BookshelfID":"8","Capacity":"750"},
    {"BookshelfID":"9","Capacity":"820"},
    {"BookshelfID":"10","Capacity":"570"}
  ],
  "products": [
    {"ProductName":"The Great Gatsby","Value":"50","Weight":"10"},
    {"ProductName":"To Kill a Mockingbird","Value":"70","Weight":"20"},
    {"ProductName":"1984","Value":"30","Weight":"5"},
    {"ProductName":"Pride and Prejudice","Value":"60","Weight":"15"},
    {"ProductName":"The Catcher in the Rye","Value":"80","Weight":"25"},
    {"ProductName":"Moby Dick","Value":"90","Weight":"30"},
    {"ProductName":"Jane Eyre","Value":"40","Weight":"12"},
    {"ProductName":"War and Peace","Value":"100","Weight":"35"},
    {"ProductName":"The Odyssey","Value":"55","Weight":"10"},
    {"ProductName":"Crime and Punishment","Value":"75","Weight":"20"},
    {"ProductName":"The Hobbit","Value":"65","Weight":"18"},
    {"ProductName":"Brave New World","Value":"95","Weight":"28"},
    {"ProductName":"Anna Karenina","Value":"45","Weight":"8"},
    {"ProductName":"Wuthering Heights","Value":"85","Weight":"22"},
    {"ProductName":"The Divine Comedy","Value":"70","Weight":"25"},
    {"ProductName":"The Iliad","Value":"110","Weight":"40"},
    {"ProductName":"Les Misérables","Value":"50","Weight":"14"},
    {"ProductName":"Dracula","Value":"60","Weight":"16"},
    {"ProductName":"Frankenstein","Value":"120","Weight":"50"},
    {"ProductName":"The Brothers Karamazov","Value":"100","Weight":"30"},
    {"ProductName":"Don Quixote","Value":"52","Weight":"11"},
    {"ProductName":"One Hundred Years of Solitude","Value":"68","Weight":"19"},
    {"ProductName":"Ulysses","Value":"38","Weight":"7"},
    {"ProductName":"The Alchemist","Value":"58","Weight":"14"},
    {"ProductName":"Meditations","Value":"82","Weight":"24"}
  ]
}