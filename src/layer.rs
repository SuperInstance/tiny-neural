use crate::{Activation, Tensor};
use rand::Rng;

pub trait Layer {
    fn forward(&mut self, input: &Tensor, training: bool) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn weights(&self) -> &Tensor;
    fn biases(&self) -> &Tensor;
    fn weights_mut(&mut self) -> &mut Tensor;
    fn biases_mut(&mut self) -> &mut Tensor;
    fn weight_grad(&self) -> &Tensor;
    fn bias_grad(&self) -> &Tensor;
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
    // Cached for backprop
    pub last_input: Option<Tensor>,
    pub last_pre_activation: Option<Tensor>,
    pub last_output: Option<Tensor>,
    pub weight_grad: Tensor,
    pub bias_grad: Tensor,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt();
        let w_vals: Vec<f64> = (0..input_size * output_size)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        DenseLayer {
            weights: Tensor::new(input_size, output_size, w_vals),
            biases: Tensor::zeros(1, output_size),
            activation,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
            weight_grad: Tensor::zeros(input_size, output_size),
            bias_grad: Tensor::zeros(1, output_size),
        }
    }

    pub fn from_weights(weights: Tensor, biases: Tensor, activation: Activation) -> Self {
        let (r, c) = weights.shape();
        DenseLayer {
            weights,
            biases,
            activation,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
            weight_grad: Tensor::zeros(r, c),
            bias_grad: Tensor::zeros(1, c),
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &Tensor, _training: bool) -> Tensor {
        self.last_input = Some(input.clone());
        // output = input × weights + biases (broadcast)
        let z = input.matmul(&self.weights).add(&self.biases.broadcast(input.rows(), self.biases.cols()));
        self.last_pre_activation = Some(z.clone());
        let a = self.activation.apply(&z);
        self.last_output = Some(a.clone());
        a
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.last_input.as_ref().expect("No cached input");
        let z = self.last_pre_activation.as_ref().expect("No cached pre-activation");

        // Activation derivative
        let act_deriv = self.activation.derivative(z);
        let delta = grad_output.mul_elem(&act_deriv);

        // Weight gradient: input^T × delta
        let w_grad = input.transpose().matmul(&delta);
        self.weight_grad = w_grad;

        // Bias gradient: sum over batch
        let mut b_grad = Tensor::zeros(1, self.biases.cols());
        for j in 0..self.biases.cols() {
            let mut s = 0.0;
            for i in 0..delta.rows() {
                s += delta.data[(i, j)];
            }
            b_grad.data[(0, j)] = s;
        }
        self.bias_grad = b_grad;

        // Input gradient: delta × weights^T
        delta.matmul(&self.weights.transpose())
    }

    fn weights(&self) -> &Tensor { &self.weights }
    fn biases(&self) -> &Tensor { &self.biases }
    fn weights_mut(&mut self) -> &mut Tensor { &mut self.weights }
    fn biases_mut(&mut self) -> &mut Tensor { &mut self.biases }
    fn weight_grad(&self) -> &Tensor { &self.weight_grad }
    fn bias_grad(&self) -> &Tensor { &self.bias_grad }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward_shape() {
        let mut layer = DenseLayer::new(3, 2, Activation::Identity);
        let input = Tensor::new(1, 3, vec![1.0, 2.0, 3.0]);
        let out = layer.forward(&input, false);
        assert_eq!(out.shape(), (1, 2));
    }

    #[test]
    fn test_dense_forward_identity() {
        let w = Tensor::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]); // identity matrix
        let b = Tensor::new(1, 2, vec![0.0, 0.0]);
        let mut layer = DenseLayer::from_weights(w, b, Activation::Identity);
        let input = Tensor::new(1, 2, vec![3.0, 4.0]);
        let out = layer.forward(&input, false);
        assert_eq!(out.as_slice(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_dense_backward_shape() {
        let mut layer = DenseLayer::new(3, 2, Activation::ReLU);
        let input = Tensor::new(1, 3, vec![1.0, 2.0, 3.0]);
        layer.forward(&input, false);
        let grad = Tensor::ones(1, 2);
        let input_grad = layer.backward(&grad);
        assert_eq!(input_grad.shape(), (1, 3));
    }

    #[test]
    fn test_dense_batch_forward() {
        let w = Tensor::new(2, 1, vec![1.0, 1.0]);
        let b = Tensor::new(1, 1, vec![0.0]);
        let mut layer = DenseLayer::from_weights(w, b, Activation::Identity);
        let input = Tensor::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let out = layer.forward(&input, false);
        assert_eq!(out.data[(0, 0)], 3.0);
        assert_eq!(out.data[(1, 0)], 7.0);
    }
}
