use crate::Tensor;

/// Supported loss functions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Loss {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
    Hinge,
}

pub trait LossFn {
    fn compute(&self, predicted: &Tensor, target: &Tensor) -> f64;
    fn gradient(&self, predicted: &Tensor, target: &Tensor) -> Tensor;
}

impl Loss {
    pub fn compute(&self, predicted: &Tensor, target: &Tensor) -> f64 {
        match self {
            Loss::MSE => {
                let diff = predicted.sub(target);
                let sq = diff.mul_elem(&diff);
                sq.mean()
            }
            Loss::CrossEntropy => {
                // predicted = softmax probabilities, target = one-hot or soft labels
                let eps = 1e-12;
                let n = predicted.rows() as f64;
                let mut loss = 0.0;
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let p = predicted.data[(i, j)].max(eps);
                        loss -= target.data[(i, j)] * p.ln();
                    }
                }
                loss / n
            }
            Loss::BinaryCrossEntropy => {
                let eps = 1e-12;
                let n = predicted.data.len() as f64;
                let mut loss = 0.0;
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let p = predicted.data[(i, j)].clamp(eps, 1.0 - eps);
                        let t = target.data[(i, j)];
                        loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
                    }
                }
                loss / n
            }
            Loss::Hinge => {
                let n = predicted.rows() as f64;
                let mut loss = 0.0;
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let margin = 1.0 - target.data[(i, j)] * predicted.data[(i, j)];
                        loss += margin.max(0.0);
                    }
                }
                loss / n
            }
        }
    }

    pub fn gradient(&self, predicted: &Tensor, target: &Tensor) -> Tensor {
        let n = predicted.data.len() as f64;
        match self {
            Loss::MSE => {
                predicted.sub(target).scale(2.0 / n)
            }
            Loss::CrossEntropy => {
                // ∂L/∂z = p - t (when p = softmax(z))
                let eps = 1e-12;
                let mut grad = Tensor::zeros(predicted.rows(), predicted.cols());
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let p = predicted.data[(i, j)].max(eps);
                        grad.data[(i, j)] = (p - target.data[(i, j)]) / predicted.rows() as f64;
                    }
                }
                grad
            }
            Loss::BinaryCrossEntropy => {
                let eps = 1e-12;
                let mut grad = Tensor::zeros(predicted.rows(), predicted.cols());
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let p = predicted.data[(i, j)].clamp(eps, 1.0 - eps);
                        let t = target.data[(i, j)];
                        grad.data[(i, j)] = (-t / p + (1.0 - t) / (1.0 - p)) / n;
                    }
                }
                grad
            }
            Loss::Hinge => {
                let mut grad = Tensor::zeros(predicted.rows(), predicted.cols());
                for i in 0..predicted.rows() {
                    for j in 0..predicted.cols() {
                        let margin = 1.0 - target.data[(i, j)] * predicted.data[(i, j)];
                        grad.data[(i, j)] = if margin > 0.0 {
                            -target.data[(i, j)] / predicted.rows() as f64
                        } else {
                            0.0
                        };
                    }
                }
                grad
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse_zero() {
        let p = Tensor::new(1, 3, vec![1.0, 2.0, 3.0]);
        let t = Tensor::new(1, 3, vec![1.0, 2.0, 3.0]);
        let loss = Loss::MSE.compute(&p, &t);
        assert_relative_eq!(loss, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_value() {
        let p = Tensor::new(1, 2, vec![1.0, 2.0]);
        let t = Tensor::new(1, 2, vec![0.0, 0.0]);
        let loss = Loss::MSE.compute(&p, &t);
        assert_relative_eq!(loss, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_gradient() {
        let p = Tensor::new(1, 2, vec![1.0, 3.0]);
        let t = Tensor::new(1, 2, vec![0.0, 1.0]);
        let g = Loss::MSE.gradient(&p, &t);
        // 2*(pred-target)/n = 2*(1)/2 = 1.0, 2*(2)/2 = 2.0
        assert_relative_eq!(g.data[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(g.data[(0, 1)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = Tensor::new(1, 3, vec![0.7, 0.2, 0.1]);
        let t = Tensor::new(1, 3, vec![1.0, 0.0, 0.0]);
        let loss = Loss::CrossEntropy.compute(&p, &t);
        assert!(loss > 0.0);
        assert_relative_eq!(loss, -0.7_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_bce_zero() {
        let p = Tensor::new(1, 2, vec![0.9, 0.1]);
        let t = Tensor::new(1, 2, vec![1.0, 0.0]);
        let loss = Loss::BinaryCrossEntropy.compute(&p, &t);
        assert!(loss < 0.3);
    }

    #[test]
    fn test_bce_gradient() {
        let p = Tensor::new(1, 1, vec![0.5]);
        let t = Tensor::new(1, 1, vec![1.0]);
        let g = Loss::BinaryCrossEntropy.gradient(&p, &t);
        // (-1/0.5) / 1 = -2.0
        assert_relative_eq!(g.data[(0, 0)], -2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hinge_loss() {
        let p = Tensor::new(1, 2, vec![1.0, -1.0]);
        let t = Tensor::new(1, 2, vec![1.0, -1.0]);
        let loss = Loss::Hinge.compute(&p, &t);
        // margin = 1 - 1*1 = 0, 1 - (-1)*(-1) = 0
        assert_relative_eq!(loss, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hinge_gradient() {
        let p = Tensor::new(1, 1, vec![0.0]);
        let t = Tensor::new(1, 1, vec![1.0]);
        let g = Loss::Hinge.gradient(&p, &t);
        // margin = 1 - 1*0 = 1 > 0, grad = -1/n = -1
        assert_relative_eq!(g.data[(0, 0)], -1.0, epsilon = 1e-10);
    }
}
