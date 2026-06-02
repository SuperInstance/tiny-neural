use nalgebra::DMatrix;

/// A 2D tensor backed by nalgebra.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: DMatrix<f64>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols, "Value count mismatch");
        Self { data: DMatrix::from_row_slice(rows, cols, &values) }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: DMatrix::zeros(rows, cols) }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self { data: DMatrix::from_element(rows, cols, 1.0) }
    }

    pub fn from_element(rows: usize, cols: usize, val: f64) -> Self {
        Self { data: DMatrix::from_element(rows, cols, val) }
    }

    pub fn rows(&self) -> usize { self.data.nrows() }
    pub fn cols(&self) -> usize { self.data.ncols() }
    pub fn shape(&self) -> (usize, usize) { (self.rows(), self.cols()) }

    /// Reshape the tensor to (rows, cols). Total element count must match.
    pub fn reshape(&self, rows: usize, cols: usize) -> Self {
        assert_eq!(self.data.len(), rows * cols, "Reshape size mismatch");
        let vals: Vec<f64> = self.data.iter().cloned().collect();
        Self { data: DMatrix::from_row_slice(rows, cols, &vals) }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch in add");
        Tensor { data: &self.data + &other.data }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch in sub");
        Tensor { data: &self.data - &other.data }
    }

    /// Element-wise multiplication.
    pub fn mul_elem(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch in mul_elem");
        Tensor { data: self.data.component_mul(&other.data) }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Tensor {
        Tensor { data: self.data.scale(s) }
    }

    /// Matrix multiplication: self (m×k) × other (k×n) → (m×n).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols(), other.rows(), "Matmul dimension mismatch");
        Tensor { data: &self.data * &other.data }
    }

    /// Transpose.
    pub fn transpose(&self) -> Tensor {
        Tensor { data: self.data.transpose() }
    }

    /// Broadcast a (1, c) or (r, 1) tensor to (rows, cols).
    pub fn broadcast(&self, rows: usize, cols: usize) -> Tensor {
        if self.shape() == (rows, cols) {
            return self.clone();
        }
        if self.rows() == 1 && self.cols() == cols {
            let mut result = DMatrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    result[(i, j)] = self.data[(0, j)];
                }
            }
            return Tensor { data: result };
        }
        if self.cols() == 1 && self.rows() == rows {
            let mut result = DMatrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    result[(i, j)] = self.data[(i, 0)];
                }
            }
            return Tensor { data: result };
        }
        panic!("Cannot broadcast {:?} to ({}, {})", self.shape(), rows, cols);
    }

    /// Apply a function element-wise.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Tensor {
        Tensor { data: self.data.map(f) }
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    /// Get a flat slice of values.
    pub fn as_slice(&self) -> Vec<f64> {
        self.data.iter().cloned().collect()
    }

    /// Create from a DMatrix directly.
    pub fn from_matrix(m: DMatrix<f64>) -> Self {
        Self { data: m }
    }

    /// Row sum (returns column vector as Tensor).
    pub fn row_sum(&self) -> Tensor {
        let mut vals = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            let mut s = 0.0;
            for j in 0..self.cols() {
                s += self.data[(i, j)];
            }
            vals.push(s);
        }
        Tensor { data: DMatrix::from_column_slice(self.rows(), 1, &vals) }
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor { self.add(other) }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor { self.sub(other) }
}
