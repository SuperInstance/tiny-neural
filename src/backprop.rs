use crate::{FeedForward, Layer, Tensor, Loss};
use crate::optimizer::Optimizer;

/// Backpropagation engine for computing gradients through the network.
pub struct BackpropEngine;

impl BackpropEngine {
    /// Run forward pass, compute loss, then backward pass.
    /// Returns (loss_value, weight_gradients_per_layer, bias_gradients_per_layer).
    pub fn compute_gradients(
        net: &mut FeedForward,
        input: &Tensor,
        target: &Tensor,
        loss_fn: Loss,
    ) -> (f64, Vec<Tensor>, Vec<Tensor>) {
        let output = net.forward(input, true);
        let loss_val = loss_fn.compute(&output, target);

        // Initial gradient from loss
        let mut grad = loss_fn.gradient(&output, target);

        let mut w_grads = Vec::new();
        let mut b_grads = Vec::new();

        // Backprop through layers in reverse
        for layer in net.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }

        // Collect gradients (in forward order)
        for layer in &net.layers {
            w_grads.push(layer.weight_grad().clone());
            b_grads.push(layer.bias_grad().clone());
        }

        (loss_val, w_grads, b_grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, DenseLayer};
    use approx::assert_relative_eq;

    #[test]
    fn test_backprop_gradient_shapes() {
        let l1 = DenseLayer::new(2, 3, Activation::ReLU);
        let l2 = DenseLayer::new(3, 1, Activation::Sigmoid);
        let mut net = FeedForward::new(vec![l1, l2]);

        let input = Tensor::new(1, 2, vec![0.5, -0.3]);
        let target = Tensor::new(1, 1, vec![1.0]);

        let (loss, w_grads, b_grads) = BackpropEngine::compute_gradients(
            &mut net, &input, &target, Loss::MSE,
        );

        assert!(loss >= 0.0);
        assert_eq!(w_grads.len(), 2);
        assert_eq!(b_grads.len(), 2);
        assert_eq!(w_grads[0].shape(), (2, 3));
        assert_eq!(w_grads[1].shape(), (3, 1));
    }

    #[test]
    fn test_backprop_reduces_loss() {
        use crate::optimizer::Sgd;

        let w = crate::Tensor::new(1, 1, vec![2.0]);
        let b = crate::Tensor::new(1, 1, vec![0.0]);
        let l = DenseLayer::from_weights(w, b, Activation::Identity);
        let mut net = FeedForward::new(vec![l]);

        let input = Tensor::new(1, 1, vec![1.0]);
        let target = Tensor::new(1, 1, vec![3.0]);

        let mut opt = Sgd { lr: 0.1 };

        let loss_before = {
            let out = net.forward(&input, false);
            Loss::MSE.compute(&out, &target)
        };

        // Train a few steps
        for _ in 0..50 {
            let (_, wg, bg) = BackpropEngine::compute_gradients(
                &mut net, &input, &target, Loss::MSE,
            );
            opt.step(&mut net, &wg, &bg);
        }

        let loss_after = {
            let out = net.forward(&input, false);
            Loss::MSE.compute(&out, &target)
        };

        assert!(loss_after < loss_before);
    }

    #[test]
    fn test_chain_rule_linear() {
        // y = w2 * (w1 * x + b1) + b2, target known
        // Verify gradient by numerical check
        let w1 = Tensor::new(1, 1, vec![1.5]);
        let b1 = Tensor::new(1, 1, vec![0.5]);
        let l1 = DenseLayer::from_weights(w1, b1, Activation::Identity);

        let w2 = Tensor::new(1, 1, vec![2.0]);
        let b2 = Tensor::new(1, 1, vec![-1.0]);
        let l2 = DenseLayer::from_weights(w2, b2, Activation::Identity);

        let mut net = FeedForward::new(vec![l1, l2]);

        let input = Tensor::new(1, 1, vec![2.0]);
        let target = Tensor::new(1, 1, vec![0.0]);

        let (_, wg, _bg) = BackpropEngine::compute_gradients(
            &mut net, &input, &target, Loss::MSE,
        );

        // y = w2*(w1*x+b1)+b2 = 2*(1.5*2+0.5)-1 = 2*3.5-1 = 6.0
        // MSE = (6-0)^2 / 1 = 36
        // dw1 = dL/dw1 = 2*(y-t) * w2 * x = 2*6*2*2 = 48
        // dw2 = 2*(y-t) * (w1*x+b1) = 2*6*3.5 = 42
        assert_relative_eq!(wg[0].data[(0, 0)], 48.0, epsilon = 1e-10);
        assert_relative_eq!(wg[1].data[(0, 0)], 42.0, epsilon = 1e-10);
    }
}
