# rust-bezier

Computes points of intersection of 2 dimentional bezier cuves of any degree. 
Calculates the resultant using a silvester matrix determinant then uses Durand–Kerner method to find roots.

## Example 
```rust 
let bezier1 = Bezier{ controllPoints : vec![
    Point { x : 4.0, y : 6.0 }, 
    Point { x : 2.0, y : 5.0 }, 
    Point {x : 2.0, y : 2.0}] };
let bezier2 = Bezier{ controllPoints : vec![
    Point { x : 2.0, y : 6.0 }, 
    Point { x : 3.0, y : 4.0 }, 
    Point {x : 6.0, y : 4.0}] };
findPointsOfIntersectionOfBeziers(bezier1, bezier2);
```
Will print out  
```
================[Sylvester Matrix]=================
2.0, -4.0, -2.0x^2 + -2.0x^1 + 2.0, 0
0, 2.0, -4.0, -2.0x^2 + -2.0x^1 + 2.0
-2.0, -2.0, -2.0x^2 + 4.0x^1 + 0.0, 0
0, -2.0, -2.0, -2.0x^2 + 4.0x^1 + 0.0
===================================================
================[Resultant]=================
0.0x^7 + 0.0x^6 + 0.0x^5 + 64.0x^4 + -64.0x^3 + 0.0x^2 + -208.0x^1 + 64.0
============================================
================[Complex Roots]=================
[Complex { re: 1.8173896145560624, im: 0.0000000000023333659621656614 }, Complex { re: -0.5595885427613132, im: -1.2288733382883859 }, Complex { re: -0.5595885427613198, im: 1.228873338288377 }, Complex { re: 0.3017874709652809, im: 0.0000000000000000000001913945687844254 }]
================================================
================[Filtered Roots]=================
t = [0.3017874709652809]
================================================
================[Points Of Intersection]=================
[Point { x: 2.785726297193802, y: 4.975001471402116 }]
=========================================================
```

