#![allow(dead_code)]
#![allow(non_snake_case)]

use num_complex::Complex;


pub trait Zero 
{
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

trait PolynomialInput: std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Sub<Output = Self> + Zero + Clone + std::fmt::Debug {}
impl<T> PolynomialInput for T where T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Sub<Output = Self> + Zero + Clone + std::fmt::Debug {}

impl Zero for i32 
{
    fn zero() -> Self
    {
        0
    }
    fn is_zero(&self) -> bool
    {
        *self == 0
    }
}

impl Zero for f64 
{
    fn zero() -> Self
    {
        0.0
    }
    fn is_zero(&self) -> bool
    {
        *self == 0.0
    }
}

impl<T : Zero> Zero for Complex<T>
{
    fn zero() -> Self
    {
        Complex::new(T::zero(), T::zero())
    }
    fn is_zero(&self) -> bool
    {
        T::is_zero(&self.re) && T::is_zero(&self.im)
    }
}

#[derive(Clone)]
struct Polynomial<T : PolynomialInput>
{
    coeficients : Vec<T>,
}

impl<T : PolynomialInput> std::ops::Add for Polynomial<T> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self 
    {
        let resultLength = std::cmp::max(self.coeficients.len(), other.coeficients.len());
        let mut result : Vec<T> = Vec::new(); 
        for i in 0..resultLength
        {
            let mut currRes = T::zero();
            if i < self.coeficients.len()
            {
                currRes = currRes + self.coeficients[i].clone();
            }
            if i < other.coeficients.len()
            {
                currRes = currRes + other.coeficients[i].clone();
            }
            result.push(currRes);
        }
        Polynomial{ coeficients : result }
    }
}

impl<T : PolynomialInput> std::ops::Sub for Polynomial<T> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self 
    {
        let resultLength = std::cmp::max(self.coeficients.len(), other.coeficients.len());
        let mut result : Vec<T> = Vec::new(); 
        for i in 0..resultLength
        {
            let mut currRes = T::zero();
            if i < self.coeficients.len()
            {
                currRes = currRes + self.coeficients[i].clone();
            }
            if i < other.coeficients.len()
            {
                currRes = currRes - other.coeficients[i].clone();
            }
            result.push(currRes);
        }
        Polynomial{ coeficients : result }
    }
}

impl<T : PolynomialInput> std::ops::Mul for Polynomial<T> {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self 
    {
        let resultLength = self.coeficients.len() + other.coeficients.len();
        let mut result : Vec<T> = Vec::new(); 
        for _ in 0..resultLength
        {
            result.push(T::zero());
        }
        for i in 0..self.coeficients.len()
        {
            for j in 0..other.coeficients.len()
            {
                result[i + j] = result[i + j].clone() + self.coeficients[i].clone() * other.coeficients[j].clone();
            }
        }
        Polynomial{ coeficients : result }
    }
}

impl<T : PolynomialInput> Zero for Polynomial<T>
{
    fn zero() -> Self
    {
        Polynomial{ coeficients : vec![] }
    }
    fn is_zero(&self) -> bool
    {
        for coeficent in self.coeficients.iter()
        {
            if !coeficent.is_zero()
            {
                return false
            }
        }
        true
    }
}

impl<T : PolynomialInput> std::fmt::Debug for Polynomial<T> 
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result 
    {
        if self.is_zero()
        {
            write!(f, "0") 
        }
        else 
        {   
            for i in (1..self.coeficients.len()).rev()
            {
                let _ = write!(f, "{:?}x^{:?} + ", self.coeficients[i], i);
            }
            write!(f, "{:?}", self.coeficients[0])
        }
        
    }
}

#[derive(Debug, Clone)]
struct Point 
{
    x : f64,
    y : f64,
}

#[derive(Debug, Clone)]
struct Bezier 
{
    controllPoints : Vec<Point>,
}

fn factorial(i : u64) -> u64
{
    if i == 0 
    {
        1
    }
    else 
    {
        i * factorial(i - 1)
    }
}   

fn evalBezier(b : Bezier, t : f64) -> Point 
{
    let n = b.controllPoints.len() - 1;
    if t == 0.0
    {
        return b.controllPoints[0].clone();
    }
    if t == 1.0
    {
        return b.controllPoints[n].clone();
    }
    let oneMinusT = 1.0 - t;
    let mut result : Point = Point { x : 0.0, y : 0.0 };
    let mut nCi = 1.0;
    let mut oneMinusTToTheNMinusI = oneMinusT.powi(n as i32);
    let mut tToTheI = 1.0;
    for i in 0..b.controllPoints.len()
    {
        result.x += (nCi as f64) * tToTheI * oneMinusTToTheNMinusI * b.controllPoints[i].x;
        result.y += (nCi as f64) * tToTheI * oneMinusTToTheNMinusI * b.controllPoints[i].y;
        tToTheI *= t;
        oneMinusTToTheNMinusI /= oneMinusT;
        nCi *= ((n - i) as f64) / ((i + 1) as f64);
    }
    result
}

fn getBezierPolynomail(b : Bezier) -> (Polynomial<f64>, Polynomial<f64>)
{
    fn C(i : usize, b : &Bezier) -> (f64, f64)
    {
        let n = b.controllPoints.len() - 1;
        let mut sumX : f64 = 0.0;
        let mut sumY : f64 = 0.0;
        for j in 0..=i
        {
            if (i + j) % 2 == 0
            {
                sumX += b.controllPoints[j].x / ((factorial(j as u64) * factorial((i - j) as u64)) as f64);
                sumY += b.controllPoints[j].y / ((factorial(j as u64) * factorial((i - j) as u64)) as f64);
            }
            else 
            {
                sumX -= b.controllPoints[j].x / ((factorial(j as u64) * factorial((i - j) as u64)) as f64);
                sumY -= b.controllPoints[j].y / ((factorial(j as u64) * factorial((i - j) as u64)) as f64);
            }
        }
        let m : f64 = (factorial(n as u64) / factorial((n - i) as u64)) as f64;
        (m * sumX, m * sumY)
    }

    let mut resultX = Vec::new();
    let mut resultY = Vec::new();
    for i in 0..b.controllPoints.len()
    {
        let (cx, cy) = C(i, &b);
        resultX.push(cx);
        resultY.push(cy);
    }
    (Polynomial { coeficients : resultX }, Polynomial { coeficients : resultY })
}


fn eval<T : PolynomialInput>(p : &Polynomial<T>, x : T) -> T
{
    let mut res : T = T::zero();
    let mut currMult : T = T::zero();
    for coeficient in p.coeficients.iter() 
    {
        if currMult.is_zero()
        {
            res = res + coeficient.clone();
            currMult = x.clone();
        }
        else 
        {
            res = res + coeficient.clone() * currMult.clone();
            currMult = currMult.clone() * x.clone();
        }
    }
    res
}

fn sylvesterMatrix<T : PolynomialInput>(p1 : &Polynomial<T>, p2 : &Polynomial<T>) -> Vec<Vec<T>>
{
    let matrixSize = p1.coeficients.len() + p2.coeficients.len() - 2;
    
    let p1Size = matrixSize - p1.coeficients.len() + 1;

    let mut matrix : Vec<Vec<T>> = Vec::new(); 
    for _ in 0..matrixSize
    {
        let mut row = Vec::new();
        for _ in 0..matrixSize
        {
            row.push(T::zero());
        }
        matrix.push(row);
    }

    for i in 0..p1Size
    {
        let startIndex = i;
        for j in 0..p1.coeficients.len()
        {
            matrix[i][startIndex + j] = p1.coeficients[p1.coeficients.len() - j - 1].clone();
        }
    }

    for i in p1Size..matrixSize
    {
        let startIndex = i - p1Size;
        for j in 0..p2.coeficients.len()
        {
            matrix[i][startIndex + j] = p2.coeficients[p2.coeficients.len() - j - 1].clone();
        }
    }

    matrix
}

fn print_matrix<T : std::fmt::Debug>(matrix : &Vec<Vec<T>>) -> ()
{
    for i in 0..matrix.len()
    {
        for j in 0..matrix.len()
        {
            print!("{:?}", matrix[i][j]);
            if j != (matrix.len() - 1)
            {
                print!(", ");
            }
        }
        println!("");
    }
}

fn getCofactor<T : PolynomialInput>(matrix : &Vec<Vec<T>>, skipX : usize, skipY : usize) -> Vec<Vec<T>>
{
    let mut result = Vec::new();
    for y in 0..matrix.len()
    {
        if y == skipY
        {
            continue;
        }
        let mut row = Vec::new();
        for x in 0..matrix[y].len()
        {
            if x == skipX
            {
                continue;
            }
            row.push(matrix[y][x].clone());
        }
        result.push(row);
    }   
    result
}

fn determinent<T : PolynomialInput>(matrix : Vec<Vec<T>>) -> T
{
    if matrix.len() == 1
    {
        return matrix[0][0].clone();
    }
    let mut sign = 1;
    let mut result : T = T::zero();
    for i in 0..matrix[0].len()
    {
        let cofactor = getCofactor(&matrix, i, 0);
        if sign == 1
        {
            result = result + matrix[0][i].clone() * determinent(cofactor);
        }
        else 
        {
            result = result - matrix[0][i].clone() * determinent(cofactor);
        }
        
        sign = -1 * sign;
    }
    result
} 

fn resultant<T : PolynomialInput>(p1 : &Polynomial<T>, p2 : &Polynomial<T>) -> T
{
    let matrix = sylvesterMatrix(p1, p2);
    println!("================[Sylvester Matrix]=================");
    print_matrix(&matrix);
    println!("===================================================");
    // compute determinent
    let det = determinent(matrix);
    det
}

fn convertToComplexPolynomial(p : Polynomial<f64>) -> Polynomial<Complex<f64>>
{
    let mut res = Polynomial { coeficients : Vec::new() };
    for i in 0..p.coeficients.len()
    {
        res.coeficients.push(Complex::new(p.coeficients[i], 0.0));
    }
    res
}

fn findRoots(p : Polynomial<Complex<f64>>) -> Vec<Complex<f64>>
{
    let mut leadingCoeficentIndex : usize = 0;
    for i in (0..p.coeficients.len()).rev()
    {
        if p.coeficients[i].is_zero()
        {
            continue;
        }
        else 
        {
            leadingCoeficentIndex = i;
            break;
        }
    }
    let mut pNormalised = Polynomial { coeficients : Vec::new() };
    for i in 0..=leadingCoeficentIndex
    {
        pNormalised.coeficients.push(p.coeficients[i] / p.coeficients[leadingCoeficentIndex]);
    }
    let mut currentGusses = Vec::new();
    let k = Complex::new(0.3, 0.7);
    let mut curr = Complex::new(1.0, 0.0);
    for _ in 0..(pNormalised.coeficients.len() - 1)
    {
        currentGusses.push(curr);
        curr *= k;
    }
    // itterate through Durand–Kerner method
    let mut cumulativeChange = 1.0;
    while cumulativeChange > 0.001
    {   
        cumulativeChange = 0.0;
        for i in 0..currentGusses.len()
        {
            let mut denomintor = Complex::new(1.0, 0.0);
            for j in 0..currentGusses.len()
            {
                if j == i
                {
                    continue;
                }
                denomintor *= currentGusses[i] - currentGusses[j];
            }
            let newGuess = currentGusses[i] - eval(&pNormalised, currentGusses[i]) / denomintor;
            cumulativeChange += (currentGusses[i] - newGuess).norm();
            currentGusses[i] = newGuess;
        }
    }
    currentGusses
}

fn findPointsOfIntersectionOfBeziers(bezier1 : Bezier, bezier2 : Bezier) -> ()
{
    let (b1xPoly, b1yPoly) = getBezierPolynomail(bezier1.clone());
    let (b2xPoly, b2yPoly) = getBezierPolynomail(bezier2.clone());

    // generate multivaraible polynomials
    let mut polyXCoeficinets = Vec::new();
    for i in 0..b1xPoly.coeficients.len()
    {
        let mut v = Vec::new();
        v.push(b1xPoly.coeficients[i]);
        let coeficientPoly = Polynomial { coeficients : v };

        if i == 0 
        {
            polyXCoeficinets.push(coeficientPoly - b2xPoly.clone());
        }
        else 
        {
            polyXCoeficinets.push(coeficientPoly);
        }
    }
    let mut polyYCoeficinets = Vec::new();
    for i in 0..b1yPoly.coeficients.len()
    {
        let mut v = Vec::new();
        v.push(b1yPoly.coeficients[i]);
        let coeficientPoly = Polynomial { coeficients : v };

        if i == 0 
        {
            polyYCoeficinets.push(coeficientPoly - b2yPoly.clone());
        }
        else 
        {
            polyYCoeficinets.push(coeficientPoly);
        }
    }
    let polyX = Polynomial { coeficients : polyXCoeficinets };
    let polyY = Polynomial { coeficients : polyYCoeficinets };

    // compute resultant
    let res = resultant(&polyX, &polyY);
    println!("================[Resultant]=================");
    println!("{:?}", res);
    println!("============================================");

    // find roots
    let roots = findRoots(convertToComplexPolynomial(res));
    println!("================[Complex Roots]=================");
    println!("{:?}", roots);
    println!("================================================");

    // filter roots
    let mut filterdRoots = Vec::new();
    for i in 0..roots.len()
    {
        if roots[i].im < 0.001 && 0.0 <= roots[i].re && roots[i].re <= 1.0
        {
            filterdRoots.push(roots[i].re);
        }
    }

    println!("================[Filterd Roots]=================");
    println!("t = {:?}", filterdRoots);
    println!("================================================");

    // find points of intersection 
    let mut pois = Vec::new();
    for i in 0..filterdRoots.len()
    {
        pois.push(evalBezier(bezier2.clone(), filterdRoots[i]));
    }

    println!("================[Points Of Intersection]=================");
    println!("{:?}", pois);
    println!("=========================================================");
}


fn main()
{
    let bezier1 = Bezier{ controllPoints : vec![
        Point { x : 4.0, y : 6.0 }, 
        Point { x : 2.0, y : 5.0 }, 
        Point {x : 2.0, y : 2.0}] };
    let bezier2 = Bezier{ controllPoints : vec![
        Point { x : 2.0, y : 6.0 }, 
        Point { x : 3.0, y : 4.0 }, 
        Point {x : 6.0, y : 4.0}] };
    findPointsOfIntersectionOfBeziers(bezier1, bezier2);
}

