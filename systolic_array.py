
from processing_element import ProcessingElement

class SystolicArray:
    """This class is our TPU simulator built using systolic array architecture. Spent a lot of time understanding Google TPUs!"""

    def __init__(self,n):
        """Set up an n x n grid of processing elements. We create all of them even if we're not fully utilizing the systolic dataflow yet."""
        # We add error checking for invalid matrix sizes
        if n <= 0:
            raise ValueError("Error: Matrix size must be positive!")
        if not isinstance(n, int):
            raise TypeError("Error: Matrix size must be an integer!")
            
        self.n=n
        self.array=[[ProcessingElement() for _ in range(n)] for _ in range(n)]  # We create a 2D grid of PEs
        print(f"  [DEBUG] Created systolic array of size {n}x{n}")

    def multiply(self,a,b):
        """Multiply two matrices using the systolic array. This is basic matrix multiplication C[i][j] = sum(a[i][k] * b[k][j])."""

        n=self.n
        
        # FIXME: We should implement true systolic dataflow but for now this basic approach works
        
        # We validate matrix sizes first - important to catch bugs early!
        try:
            if len(a) != n or len(b) != n:
                raise ValueError(f"Error: Matrices must be {n}x{n}!")
            if len(a[0]) != n or len(b[0]) != n:
                raise ValueError(f"Error: Matrices must be square {n}x{n}!")
        except TypeError:
            raise TypeError("Error: Matrices must be 2D arrays!")
        
        c=[[0]*n for _ in range(n)]  # We initialize result matrix c here
        
        print(f"  [DEBUG] Multiplying {n}x{n} matrices...")

        # We iterate through rows and columns manually (not using fancy numpy)
        for i in range(n):  # row iterator for matrix a
            for j in range(n):  # column iterator for matrix b
                # We accumulate the product for position (i,j)
                for k in range(n):  # multiply and sum k times
                    c[i][j] += a[i][k] * b[k][j]
                
        print(f"  [DEBUG] Multiplication completed successfully")
        return c
