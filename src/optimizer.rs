use crate::{FeedForward, Layer, Tensor};

/// Optimizer trait — all optimizers implement this.
pub trait Optimizer {
    fn step(
        &mut self,
        net: &mut FeedForward,
        weight_grads: &[Tensor],
        bias_grads: &[Tensor],
    );
}

/// Vanilla SGD optimizer.
pub struct Sgd {
    pub lr: f64,
}

impl Optimizer for Sgd {
    fn step(
        &mut self,
        net: &mut FeedForward,
        weight_grads: &[Tensor],
        bias_grads: &[Tensor],
    ) {
        for (i, layer) in net.layers.iter_mut().enumerate() {
            let w_update = weight_grads[i].scale(-self.lr);
            let b_update = bias_grads[i].scale(-self.lr);
            *layer.weights_mut() = layer.weights().add(&w_update);
            *layer.biases_mut() = layer.biases().add(&b_update);
        }
    }
}

/// SGD with momentum.
pub struct SgdMomentum {
    pub lr: f64,
    pub momentum: f64,
    velocity_w: Vec<Tensor>,
    velocity_b: Vec<Tensor>,
}

impl SgdMomentum {
    pub fn new(lr: f64, momentum: f64, num_layers: usize) -> Self {
        Self {
            lr,
            momentum,
            velocity_w: Vec::new(),
            velocity_b: Vec::new(),
        }
    }
}

impl Optimizer for SgdMomentum {
    fn step(
        &mut self,
        net: &mut FeedForward,
        weight_grads: &[Tensor],
        bias_grads: &[Tensor],
    ) {
        if self.velocity_w.len() != net.layers.len() {
            self.velocity_w = net.layers.iter().map(|l| Tensor::zeros_from(l.weights())).collect();
            self.velocity_b = net.layers.iter().map(|l| Tensor::zeros_from(l.biases())).collect();
        }

        for (i, layer) in net.layers.iter_mut().enumerate() {
            self.velocity_w[i] = self.velocity_w[i]
                .scale(self.momentum)
                .add(&weight_grads[i].scale(-self.lr));
            self.velocity_b[i] = self.velocity_b[i]
                .scale(self.momentum)
                .add(&bias_grads[i].scale(-self.lr));

            *layer.weights_mut() = layer.weights().add(&self.velocity_w[i]);
            *layer.biases_mut() = layer.biases().add(&self.velocity_b[i]);
        }
    }
}

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    m_w: Vec<Tensor>,
    v_w: Vec<Tensor>,
    m_b: Vec<Tensor>,
    v_b: Vec<Tensor>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w: Vec::new(),
            v_w: Vec::new(),
            m_b: Vec::new(),
            v_b: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        net: &mut FeedForward,
        weight_grads: &[Tensor],
        bias_grads: &[Tensor],
    ) {
        if self.m_w.len() != net.layers.len() {
            self.m_w = net.layers.iter().map(|l| Tensor::zeros_from(l.weights())).collect();
            self.v_w = net.layers.iter().map(|l| Tensor::zeros_from(l.weights())).collect();
            self.m_b = net.layers.iter().map(|l| Tensor::zeros_from(l.biases())).collect();
            self.v_b = net.layers.iter().map(|l| Tensor::zeros_from(l.biases())).collect();
        }

        self.t += 1;

        for (i, layer) in net.layers.iter_mut().enumerate() {
            // Weights
            let (mw, vw) = adam_update(
                &self.m_w[i], &self.v_w[i], &weight_grads[i],
                self.beta1, self.beta2, self.t,
            );
            self.m_w[i] = mw;
            self.v_w[i] = vw;

            let bias_corrected_m = self.m_w[i].scale(1.0 / (1.0 - self.beta1.powi(self.t as i32)));
            let bias_corrected_v = self.v_w[i].scale(1.0 / (1.0 - self.beta2.powi(self.t as i32)));

            let mut update = Tensor::zeros_from(layer.weights());
            for k in 0..update.data.nrows() {
                for j in 0..update.data.ncols() {
                    update.data[(k, j)] = -self.lr * bias_corrected_m.data[(k, j)]
                        / (bias_corrected_v.data[(k, j)].sqrt() + self.epsilon);
                }
            }
            *layer.weights_mut() = layer.weights().add(&update);

            // Biases
            let (mb, vb) = adam_update(
                &self.m_b[i], &self.v_b[i], &bias_grads[i],
                self.beta1, self.beta2, self.t,
            );
            self.m_b[i] = mb;
            self.v_b[i] = vb;

            let bias_corrected_mb = self.m_b[i].scale(1.0 / (1.0 - self.beta1.powi(self.t as i32)));
            let bias_corrected_vb = self.v_b[i].scale(1.0 / (1.0 - self.beta2.powi(self.t as i32)));

            let mut update_b = Tensor::zeros_from(layer.biases());
            for k in 0..update_b.data.nrows() {
                for j in 0..update_b.data.ncols() {
                    update_b.data[(k, j)] = -self.lr * bias_corrected_mb.data[(k, j)]
                        / (bias_corrected_vb.data[(k, j)].sqrt() + self.epsilon);
                }
            }
            *layer.biases_mut() = layer.biases().add(&update_b);
        }
    }
}

fn adam_update(
    m: &Tensor, v: &Tensor, grad: &Tensor,
    beta1: f64, beta2: f64, _t: usize,
) -> (Tensor, Tensor) {
    let new_m = m.scale(beta1).add(&grad.scale(1.0 - beta1));
    let new_v = v.scale(beta2).add(&grad.mul_elem(grad).scale(1.0 - beta2));
    (new_m, new_v)
}

/// Helper to create a zero tensor with the same shape.
impl Tensor {
    pub fn zeros_from(other: &Tensor) -> Self {
        Tensor::zeros(other.rows(), other.cols())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, DenseLayer, Loss, BackpropEngine};
    use approx::assert_relative_eq;

    fn simple_net() -> FeedForward {
        let w = Tensor::new(1, 1, vec![0.5]);
        let b = Tensor::new(1, 1, vec![0.0]);
        FeedForward::new(vec![DenseLayer::from_weights(w, b, Activation::Identity)])
    }

    #[test]
    fn test_sgd_convergence() {
        let mut net = simple_net();
        let input = Tensor::new(1, 1, vec![1.0]);
        let target = Tensor::new(1, 1, vec![2.0]);
        let mut opt = Sgd { lr: 0.1 };

        for _ in 0..100 {
            let (_, wg, bg) = BackpropEngine::compute_gradients(
                &mut net, &input, &target, Loss::MSE,
            );
            opt.step(&mut net, &wg, &bg);
        }

        let out = net.forward(&input, false);
        assert_relative_eq!(out.data[(0, 0)], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_sgd_momentum_convergence() {
        let mut net = simple_net();
        let input = Tensor::new(1, 1, vec![1.0]);
        let target = Tensor::new(1, 1, vec![2.0]);
        let mut opt = SgdMomentum::new(0.1, 0.9, 1);

        for _ in 0..100 {
            let (_, wg, bg) = BackpropEngine::compute_gradients(
                &mut net, &input, &target, Loss::MSE,
            );
            opt.step(&mut net, &wg, &bg);
        }

        let out = net.forward(&input, false);
        assert_relative_eq!(out.data[(0, 0)], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_adam_convergence() {
        let mut net = simple_net();
        let input = Tensor::new(1, 1, vec![1.0]);
        let target = Tensor::new(1, 1, vec![2.0]);
        let mut opt = Adam::new(0.1);

        for _ in 0..200 {
            let (_, wg, bg) = BackpropEngine::compute_gradients(
                &mut net, &input, &target, Loss::MSE,
            );
            opt.step(&mut net, &wg, &bg);
        }

        let out = net.forward(&input, false);
        assert_relative_eq!(out.data[(0, 0)], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_adam_multilayer_convergence() {
        let w1 = Tensor::new(1, 4, vec![0.1, 0.2, -0.1, 0.3]);
        let b1 = Tensor::new(1, 4, vec![0.0; 4]);
        let l1 = DenseLayer::from_weights(w1, b1, Activation::ReLU);

        let w2 = Tensor::new(4, 1, vec![0.1, -0.1, 0.2, 0.05]);
        let b2 = Tensor::new(1, 1, vec![0.0]);
        let l2 = DenseLayer::from_weights(w2, b2, Activation::Identity);

        let mut net = FeedForward::new(vec![l1, l2]);
        let input = Tensor::new(1, 1, vec![1.0]);
        let target = Tensor::new(1, 1, vec![1.0]);
        let mut opt = Adam::new(0.01);

        let initial_loss = {
            let out = net.forward(&input, false);
            Loss::MSE.compute(&out, &target)
        };

        for _ in 0..300 {
            let (_, wg, bg) = BackpropEngine::compute_gradients(
                &mut net, &input, &target, Loss::MSE,
            );
            opt.step(&mut net, &wg, &bg);
        }

        let final_loss = {
            let out = net.forward(&input, false);
            Loss::MSE.compute(&out, &target)
        };

        assert!(final_loss < initial_loss);
    }
}
