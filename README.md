# tiny-neural

Neural networks from scratch in Rust. No CUDA, no frameworks. Just tensors and gradients.

A from-scratch neural network library: tensors, activations, loss functions, dense layers, backpropagation, optimizers (SGD, SGD+momentum, Adam), regularization (dropout, batch norm), weight initialization (Xavier, He), convolution (1D, 2D, pooling), and a neural decision agent.

Built on `nalgebra` for linear algebra, `serde` for serialization, and `rand` for stochastic operations. No framework dependencies — every layer of the network is implemented explicitly.

## Features

- **Tensors** — 2D matrix type wrapping `nalgebra::DMatrix<f64>`, with arithmetic ops, matmul, broadcasting, reduction, and element-wise transforms.
- **Activations** — Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, GELU, and Identity — each with a verified `derivative()` for backprop.
- **Loss functions** — MSE, Cross-Entropy, Binary Cross-Entropy, and Hinge loss — each with analytic gradients.
- **Layers** — `DenseLayer` (fully connected) with forward/backward, caching pre-activations for clean gradient computation.
- **Networks** — `FeedForward` chains layers sequentially; `BackpropEngine` runs forward → loss → reverse-mode backprop in one call.
- **Optimizers** — Vanilla SGD, SGD with momentum, and Adam (bias-corrected first/second moment estimates).
- **Regularization** — `Dropout` (inverted, training-mode only) and `BatchNorm1D` (learnable γ/β, running stats, full backward pass).
- **Initialization** — Xavier uniform, Xavier normal, and He/Kaiming normal.
- **Convolution** — `Conv1D`, `Conv2D`, and `Pool2D` (max or average).
- **Agent** — `NeuralAgent` wraps a policy network that observes state tensors and selects actions.

## Install

```toml
[dependencies]
tiny-neural = "0.1"
```

## Quick Start

### Train a network to fit y = 2x

```rust
use tiny_neural::*;

let w = Tensor::new(1, 1, vec![0.5]);
let b = Tensor::new(1, 1, vec![0.0]);
let layer = DenseLayer::from_weights(w, b, Activation::Identity);
let mut net = FeedForward::new(vec![layer]);

let input = Tensor::new(1, 1, vec![1.0]);
let target = Tensor::new(1, 1, vec![2.0]);

let mut opt = optimizer::Adam::new(0.1);

for epoch in 0..200 {
    let (loss, wg, bg) = BackpropEngine::compute_gradients(
        &mut net, &input, &target, Loss::MSE,
    );
    opt.step(&mut net, &wg, &bg);
}

let output = net.forward(&input, false);
println!("predicted: {:.4} (target: 2.0)", output.data[(0, 0)]);
```

### Neural agent for decision-making

```rust
use tiny_neural::*;

let mut agent = NeuralAgent::new(3, 8, vec!["left".into(), "right".into(), "forward".into()]);
let obs = Tensor::new(1, 3, vec![0.1, 0.9, -0.2]);
let action = agent.act(&obs);
println!("agent chose: {}", agent.action_label(action));
```

## Tests

58 inline unit tests:

```bash
cargo test
```

## License

MIT OR Apache-2.0
