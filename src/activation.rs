use crate::Tensor;

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f64), // negative slope
    Softmax,
    GELU,
    Identity,
}

pub trait ActivationFn {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&self, input: &Tensor) -> Tensor; // derivative w.r.t. input
}

impl Activation {
    pub fn apply(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => input.map(sigmoid),
            Activation::Tanh => input.map(|x| x.tanh()),
            Activation::ReLU => input.map(|x| x.max(0.0)),
            Activation::LeakyReLU(alpha) => input.map(|x| if x > 0.0 { x } else { alpha * x }),
            Activation::Softmax => softmax(input),
            Activation::GELU => input.map(gelu),
            Activation::Identity => input.clone(),
        }
    }

    /// Returns the derivative of the activation, evaluated at the pre-activation input.
    pub fn derivative(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => {
                let s = input.map(sigmoid);
                s.mul_elem(&s.map(|x| 1.0 - x))
            }
            Activation::Tanh => input.map(|x| 1.0 - x.tanh().powi(2)),
            Activation::ReLU => input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU(alpha) => {
                input.map(|x| if x > 0.0 { 1.0 } else { *alpha })
            }
            Activation::Softmax => {
                // For softmax + cross-entropy combined, derivative simplifies.
                // Here we return ones; the combined derivative is handled in loss.
                softmax(input).map(|_| 1.0)
            }
            Activation::GELU => input.map(gelu_deriv),
            Activation::Identity => Tensor::ones(input.rows(), input.cols()),
        }
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax(input: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(input.rows(), input.cols());
    for i in 0..input.rows() {
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..input.cols() {
            max_val = max_val.max(input.data[(i, j)]);
        }
        let mut exps = Vec::with_capacity(input.cols());
        let mut sum = 0.0;
        for j in 0..input.cols() {
            let e = (input.data[(i, j)] - max_val).exp();
            exps.push(e);
            sum += e;
        }
        for j in 0..input.cols() {
            result.data[(i, j)] = exps[j] / sum;
        }
    }
    result
}

pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (std::f64::consts::SQRT_2 * x).tanh())
    // Approximation: x * Φ(x) using tanh approximation
}

pub fn gelu_deriv(x: f64) -> f64 {
    let sqrt2 = std::f64::consts::SQRT_2;
    let inner = sqrt2 * x;
    let tanh_val = inner.tanh();
    let sech2 = 1.0 - tanh_val * tanh_val;
    0.5 * (1.0 + tanh_val) + x * sqrt2 * 0.5 * sech2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid() {
        let t = Tensor::new(1, 3, vec![0.0, 1.0, -1.0]);
        let out = Activation::Sigmoid.apply(&t);
        assert_relative_eq!(out.data[(0, 0)], 0.5, epsilon = 1e-10);
        assert!(out.data[(0, 1)] > 0.5);
        assert!(out.data[(0, 2)] < 0.5);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let t = Tensor::new(1, 3, vec![0.0, 100.0, -100.0]);
        let d = Activation::Sigmoid.derivative(&t);
        // At 0, sigmoid'(0) = 0.25
        assert_relative_eq!(d.data[(0, 0)], 0.25, epsilon = 1e-10);
        // At extremes, derivative → 0
        assert!(d.data[(0, 1)] < 0.01);
        assert!(d.data[(0, 2)] < 0.01);
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::new(1, 2, vec![0.0, 1.0]);
        let out = Activation::Tanh.apply(&t);
        assert_relative_eq!(out.data[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(out.data[(0, 1)], 1.0_f64.tanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_tanh_derivative() {
        let t = Tensor::new(1, 1, vec![0.0]);
        let d = Activation::Tanh.derivative(&t);
        assert_relative_eq!(d.data[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::new(1, 4, vec![-2.0, -0.5, 0.0, 3.0]);
        let out = Activation::ReLU.apply(&t);
        assert_relative_eq!(out.data[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(out.data[(0, 3)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_relu_derivative() {
        let t = Tensor::new(1, 3, vec![-1.0, 0.0, 2.0]);
        let d = Activation::ReLU.derivative(&t);
        assert_relative_eq!(d.data[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d.data[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d.data[(0, 2)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::new(1, 2, vec![-1.0, 2.0]);
        let out = Activation::LeakyReLU(0.01).apply(&t);
        assert_relative_eq!(out.data[(0, 0)], -0.01, epsilon = 1e-10);
        assert_relative_eq!(out.data[(0, 1)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let t = Tensor::new(1, 2, vec![-1.0, 2.0]);
        let d = Activation::LeakyReLU(0.01).derivative(&t);
        assert_relative_eq!(d.data[(0, 0)], 0.01, epsilon = 1e-10);
        assert_relative_eq!(d.data[(0, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Tensor::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let out = Activation::Softmax.apply(&t);
        let sum: f64 = out.as_slice().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_batch() {
        let t = Tensor::new(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let out = Activation::Softmax.apply(&t);
        let sum0: f64 = (0..3).map(|j| out.data[(0, j)]).sum();
        let sum1: f64 = (0..3).map(|j| out.data[(1, j)]).sum();
        assert_relative_eq!(sum0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::new(1, 1, vec![0.0]);
        let out = Activation::GELU.apply(&t);
        assert_relative_eq!(out.data[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gelu_derivative_at_zero() {
        let t = Tensor::new(1, 1, vec![0.0]);
        let d = Activation::GELU.derivative(&t);
        assert_relative_eq!(d.data[(0, 0)], 0.5, epsilon = 1e-6);
    }
}
