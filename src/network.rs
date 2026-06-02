use crate::{DenseLayer, Layer, Tensor};

/// A simple feed-forward neural network.
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub layers: Vec<DenseLayer>,
}

impl FeedForward {
    pub fn new(layers: Vec<DenseLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass: returns the output tensor.
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Tensor {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current, training);
        }
        current
    }

    /// Get all weight tensors (read-only).
    pub fn weights(&self) -> Vec<&Tensor> {
        self.layers.iter().map(|l| l.weights()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Activation;

    #[test]
    fn test_network_single_layer() {
        let layer = DenseLayer::new(2, 1, Activation::Identity);
        let mut net = FeedForward::new(vec![layer]);
        let input = Tensor::new(1, 2, vec![1.0, 1.0]);
        let out = net.forward(&input, false);
        assert_eq!(out.cols(), 1);
    }

    #[test]
    fn test_network_two_layers() {
        let l1 = DenseLayer::new(2, 4, Activation::ReLU);
        let l2 = DenseLayer::new(4, 1, Activation::Sigmoid);
        let mut net = FeedForward::new(vec![l1, l2]);
        let input = Tensor::new(1, 2, vec![0.5, -0.5]);
        let out = net.forward(&input, false);
        assert_eq!(out.cols(), 1);
        // Sigmoid output in [0, 1]
        assert!(out.data[(0, 0)] > 0.0 && out.data[(0, 0)] < 1.0);
    }
}
