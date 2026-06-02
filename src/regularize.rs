use crate::Tensor;
use rand::Rng;

/// Dropout layer — randomly zeros elements during training.
#[derive(Debug, Clone)]
pub struct Dropout {
    pub rate: f64,
    pub mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        assert!((0.0..1.0).contains(&rate), "Dropout rate must be in [0, 1)");
        Self { rate, mask: None }
    }

    /// Apply dropout during training. During inference (training=false), returns input unchanged.
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        if !training || self.rate == 0.0 {
            self.mask = None;
            return input.clone();
        }
        let mut rng = rand::thread_rng();
        let keep_prob = 1.0 - self.rate;
        let mut mask_data = Vec::with_capacity(input.rows() * input.cols());
        for _ in 0..(input.rows() * input.cols()) {
            let kept = rng.gen::<f64>() >= self.rate;
            mask_data.push(if kept { 1.0 / keep_prob } else { 0.0 });
        }
        let mask = Tensor::new(input.rows(), input.cols(), mask_data);
        let output = input.mul_elem(&mask);
        self.mask = Some(mask);
        output
    }

    /// Backward pass: gradient flows only through kept units.
    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        match &self.mask {
            Some(mask) => grad_output.mul_elem(mask),
            None => grad_output.clone(),
        }
    }
}

/// Basic Batch Normalization for 1D (fully connected) layers.
/// Normalizes across features for each sample independently.
#[derive(Debug, Clone)]
pub struct BatchNorm1D {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
    pub momentum: f64,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    // Cached for backward
    pub cached_normalized: Option<Tensor>,
    pub cached_input: Option<Tensor>,
    pub cached_mean: Option<Tensor>,
    pub cached_var: Option<Tensor>,
    // Gradients
    pub gamma_grad: Tensor,
    pub beta_grad: Tensor,
}

impl BatchNorm1D {
    pub fn new(num_features: usize) -> Self {
        BatchNorm1D {
            gamma: Tensor::ones(1, num_features),
            beta: Tensor::zeros(1, num_features),
            eps: 1e-5,
            momentum: 0.1,
            running_mean: Tensor::zeros(1, num_features),
            running_var: Tensor::ones(1, num_features),
            cached_normalized: None,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            gamma_grad: Tensor::zeros(1, num_features),
            beta_grad: Tensor::zeros(1, num_features),
        }
    }

    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        let n = input.rows() as f64;
        let mean = input.mean_axis(0); // mean over batch dimension
        let diff = input.sub(&mean.broadcast(input.rows(), input.cols()));
        let var = diff.mul_elem(&diff).sum_axis(0).scale(1.0 / n);

        if training {
            // Update running stats
            let one_minus_m = 1.0 - self.momentum;
            self.running_mean = self.running_mean.scale(one_minus_m).add(&mean.scale(self.momentum));
            self.running_var = self.running_var.scale(one_minus_m).add(&var.scale(self.momentum));
        }

        let (m, v) = if training {
            (mean.clone(), var.clone())
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        let normalized = diff.mul_elem(
            &v.map(|x| 1.0 / (x + self.eps).sqrt()).broadcast(input.rows(), input.cols())
        );

        self.cached_input = Some(input.clone());
        self.cached_mean = Some(m.clone());
        self.cached_var = Some(v.clone());
        self.cached_normalized = Some(normalized.clone());

        // Scale and shift
        let gamma_broad = self.gamma.broadcast(input.rows(), input.cols());
        let beta_broad = self.beta.broadcast(input.rows(), input.cols());
        normalized.mul_elem(&gamma_broad).add(&beta_broad)
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.cached_input.as_ref().unwrap();
        let normalized = self.cached_normalized.as_ref().unwrap();
        let _mean = self.cached_mean.as_ref().unwrap();
        let var = self.cached_var.as_ref().unwrap();
        let n = input.rows() as f64;

        let gamma_broad = self.gamma.broadcast(input.rows(), input.cols());

        // dL/dgamma = sum(dL/dy * x_hat)
        self.gamma_grad = grad_output.mul_elem(normalized).sum_axis(0);
        // dL/dbeta = sum(dL/dy)
        self.beta_grad = grad_output.sum_axis(0);

        // dL/dx (simplified for batch norm)
        let dx_hat = grad_output.mul_elem(&gamma_broad);
        let std_inv = var.map(|x| 1.0 / (x + self.eps).sqrt()).broadcast(input.rows(), input.cols());
        let dx = dx_hat
            .sub(&dx_hat.sum_axis(0).broadcast(input.rows(), input.cols()).scale(1.0 / n))
            .sub(&normalized.mul_elem(&dx_hat).sum_axis(0).broadcast(input.rows(), input.cols()).scale(1.0 / n).mul_elem(normalized))
            .mul_elem(&std_inv);

        dx
    }
}

impl Tensor {
    /// Mean over rows (batch dimension), returns (1, cols) tensor.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        if axis == 0 {
            let mut vals = Vec::with_capacity(self.cols());
            for j in 0..self.cols() {
                let mut s = 0.0;
                for i in 0..self.rows() {
                    s += self.data[(i, j)];
                }
                vals.push(s / self.rows() as f64);
            }
            Tensor::new(1, self.cols(), vals)
        } else {
            let mut vals = Vec::with_capacity(self.rows());
            for i in 0..self.rows() {
                let mut s = 0.0;
                for j in 0..self.cols() {
                    s += self.data[(i, j)];
                }
                vals.push(s / self.cols() as f64);
            }
            Tensor::new(self.rows(), 1, vals)
        }
    }

    /// Sum over rows (batch dimension), returns (1, cols) tensor.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        if axis == 0 {
            let mut vals = Vec::with_capacity(self.cols());
            for j in 0..self.cols() {
                let mut s = 0.0;
                for i in 0..self.rows() {
                    s += self.data[(i, j)];
                }
                vals.push(s);
            }
            Tensor::new(1, self.cols(), vals)
        } else {
            let mut vals = Vec::with_capacity(self.rows());
            for i in 0..self.rows() {
                let mut s = 0.0;
                for j in 0..self.cols() {
                    s += self.data[(i, j)];
                }
                vals.push(s);
            }
            Tensor::new(self.rows(), 1, vals)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dropout_training_zeros_some() {
        let mut dropout = Dropout::new(0.5);
        let input = Tensor::ones(1, 100);
        let output = dropout.forward(&input, true);
        let zeros = output.as_slice().iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0, "Dropout should zero some elements");
    }

    #[test]
    fn test_dropout_inference_no_zeros() {
        let mut dropout = Dropout::new(0.5);
        let input = Tensor::ones(1, 10);
        let output = dropout.forward(&input, false);
        assert_eq!(output.as_slice(), input.as_slice());
    }

    #[test]
    fn test_dropout_backward() {
        let mut dropout = Dropout::new(0.5);
        let input = Tensor::ones(1, 10);
        let _ = dropout.forward(&input, true);
        let grad = Tensor::ones(1, 10);
        let grad_out = dropout.backward(&grad);
        // Some gradients should be 0 where dropout was applied
        let zeros = grad_out.as_slice().iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0);
    }

    #[test]
    fn test_batchnorm_forward_shape() {
        let mut bn = BatchNorm1D::new(3);
        let input = Tensor::new(4, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        let out = bn.forward(&input, true);
        assert_eq!(out.shape(), (4, 3));
    }

    #[test]
    fn test_batchnorm_normalizes() {
        let mut bn = BatchNorm1D::new(2);
        let input = Tensor::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let out = bn.forward(&input, true);
        // Output should be roughly normalized — check column means are near 0
        let mean = out.mean_axis(0);
        assert_relative_eq!(mean.data[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean.data[(0, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batchnorm_backward_shape() {
        let mut bn = BatchNorm1D::new(3);
        let input = Tensor::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        bn.forward(&input, true);
        let grad = Tensor::ones(2, 3);
        let dx = bn.backward(&grad);
        assert_eq!(dx.shape(), (2, 3));
    }
}
