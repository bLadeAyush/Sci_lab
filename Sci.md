#Bisection
```
function root = bisection(f, a, b, tol, max_iter)
    if f(a) * f(b) >= 0 then
        error("Invalid interval. f(a) and f(b) must have opposite signs.");
    end
    
    iter = 0;
    while (b - a) / 2 > tol & iter < max_iter do
        c = (a + b) / 2;
        if f(c) == 0 then
            root = c;
            return;
        elseif f(a) * f(c) < 0 then
            b = c;
        else
            a = c;
        end
        iter = iter + 1;
    end
    
    root = (a + b) / 2;
end

// Example usage
function y = f(x)
    y = x^3 - x - 2;
end

a = 1;
b = 2;
tol = 1e-6;
max_iter = 100;

root = bisection(f, a, b, tol, max_iter);
disp("Root:");
disp(root);
```



#Regula falsi
```
function root = regula_falsi(f, a, b, tol, max_iter)
    if f(a) * f(b) >= 0 then
        error("Invalid interval. f(a) and f(b) must have opposite signs.");
    end
    
    iter = 0;
    c = a;
    while abs(f(c)) > tol & iter < max_iter do
        c = (a * f(b) - b * f(a)) / (f(b) - f(a));
        if f(c) == 0 then
            root = c;
            return;
        elseif f(a) * f(c) < 0 then
            b = c;
        else
            a = c;
        end
        iter = iter + 1;
    end
    
    root = c;
end

// Example usage
function y = f(x)
    y = x^3 - x - 2;
end

a = 1;
b = 2;
tol = 1e-6;
max_iter = 100;

root = regula_falsi(f, a, b, tol, max_iter);
disp("Root:");
disp(root);
```


#NewtonRaphson 
```
function root = newton_raphson(f, df, x0, tol, max_iter)
    iter = 0;
    x = x0;
    while abs(f(x)) > tol & iter < max_iter do
        if df(x) == 0 then
            error("Zero derivative encountered. Method fails.");
        end
        x = x - f(x) / df(x);
        iter = iter + 1;
    end
    root = x;
end

// Example usage
function y = f(x)
    y = x^3 - x - 2;
end

function dy = df(x)
    dy = 3*x^2 - 1;
end

x0 = 1.5;
tol = 1e-6;
max_iter = 100;

root = newton_raphson(f, df, x0, tol, max_iter);
disp("Root:");
disp(root);
```

#Gauss
```
function [A, b] = gaussian_elimination(A, b)
    [m, n] = size(A);  // Get the dimensions of the matrix A
    rank_A = 0;        // Initialize the rank of the matrix

    // Forward elimination to get row echelon form
    for k = 1:min(m, n)
        // Find the pivot element
        [maxval, pivot_row] = max(abs(A(k:m, k)));
        pivot_row = pivot_row + k - 1;

        if maxval < 1e-10 then
            continue;  // Skip if the pivot is zero (or nearly zero)
        end

        // Swap rows if necessary
        if pivot_row ~= k then
            A([k, pivot_row], :) = A([pivot_row, k], :);
            b([k, pivot_row]) = b([pivot_row, k]);
        end

        // Normalize the pivot row
        pivot = A(k, k);
        A(k, :) = A(k, :) / pivot;
        b(k) = b(k) / pivot;

        // Eliminate the current column in the rows below
        for i = k+1:m
            factor = A(i, k);
            A(i, :) = A(i, :) - factor * A(k, :);
            b(i) = b(i) - factor * b(k);
        end

        rank_A = rank_A + 1;  // Increment the rank
    end

    // Backward substitution to get reduced row echelon form (RREF)
    for k = rank_A:-1:1
        for i = k-1:-1:1
            factor = A(i, k);
            A(i, :) = A(i, :) - factor * A(k, :);
            b(i) = b(i) - factor * b(k);
        end
    end
endfunction

// Example usage
A = [-2, 10, -1, -1; 
     10, -2, -1, -1;
     -1, -1, -2, 10;
     -1, -1, 10, -2];

b = [15; 3; -9; 27];

[A_rref, b_rref] = gaussian_elimination(A, b);  // Call the correct function name

disp("Reduced Row Echelon Form of A:");
disp(A_rref);
disp("Transformed b:");
disp(b_rref);
```

#Gauss Seidel
```
function x = gauss_seidel(A, b, tol, max_iter)
    n = length(b);
    x_old = zeros(n, 1);
    iter = 0;
    for i = 1:n
        max_row = i;
        for j = i+1:n
            if abs(A(j,i)) > abs(A(max_row,i))
                max_row = j;
            end
        end
        
        if max_row ~= i
            temp = A(i,:);
            A(i,:) = A(max_row,:);
            A(max_row,:) = temp;
            
            temp_b = b(i);
            b(i) = b(max_row);
            b(max_row) = temp_b;
        end
  end
    while iter < max_iter
        for i = 1:n
            sum = 0;
            for j = 1:n
                if i ~= j
                    sum = sum + A(i,j) * x_old(j); 
                end
            end
            
            if A(i,i) == 0
                error("Zero found on diagonal, method fails.");
            end
            
            x_old(i) = (b(i) - sum) / A(i,i); 
        end
   
        if A*x_old - b < tol
            break;
        end
        
        iter = iter + 1;
    end
    
    x = x_old;
endfunction
A = [-2,10,-1,-1; 
     10,-2,-1,-1;
     -1,-1,-2,10;
     -1,-1,10,-2];
     
b = [15; 3; -9; 27];

tol = 1e-6;  
max_iter = 100; 

x = gauss_seidel(A, b, tol, max_iter);
disp("Solution:");
disp(x);
```

#Gauss Jacobi
```
function x = gauss_jacobi(A, b, tol, max_iter)
    n = length(b);
    x_old = zeros(n, 1);
    x_new = x_old;
    iter = 0;
    for i = 1:n
        max_row = i;
        for j = i+1:n
            if abs(A(j,i)) > abs(A(max_row,i))
                max_row = j;
            end
        end
        
        if max_row ~= i
            temp = A(i,:);
            A(i,:) = A(max_row,:);
            A(max_row,:) = temp;
            
            temp_b = b(i);
            b(i) = b(max_row);
            b(max_row) = temp_b;
        end
    end
    
    while iter < max_iter
        for i = 1:n
            sum = 0;
            for j = 1:n
                if i ~= j
                    sum = sum + A(i,j) * x_old(j);
                end
            end
            x_new(i) = (b(i) - sum) / A(i,i);
        end
        
        if x_new - x_old < tol
            break;
        end
        
        x_old = x_new;
        iter = iter + 1;
    end
    
    x = x_new;
endfunction

A = [-2,10,-1,-1; 
     10,-2,-1,-1;
     -1,-1,-2,10;
     -1,-1,10,-2];
     
b = [15; 3; -9; 27];

tol = 1e-6;  
max_iter = 100; 

x = gauss_jacobi(A, b, tol, max_iter)
disp("Solution:")
disp(x)
```



