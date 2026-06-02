use crate::Tensor;

/// 1D convolution layer.
#[derive(Debug, Clone)]
pub struct Conv1D {
    pub kernel: Tensor, // (kernel_size, in_channels)
    pub stride: usize,
    pub padding: usize,
}

impl Conv1D {
    pub fn new(in_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let scale = (2.0 / (kernel_size * in_channels) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let vals: Vec<f64> = (0..kernel_size * in_channels)
            .map(|_| rand::Rng::gen_range(&mut rng, -scale..scale))
            .collect();
        Conv1D {
            kernel: Tensor::new(kernel_size, in_channels, vals),
            stride,
            padding,
        }
    }

    /// Input shape: (length, channels). Output shape: (output_length, channels).
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let (seq_len, channels) = input.shape();
        let kernel_size = self.kernel.rows();
        let output_len = (seq_len + 2 * self.padding - kernel_size) / self.stride + 1;

        let mut output = Tensor::zeros(output_len, channels);

        for i in 0..output_len {
            let start = i * self.stride;
            for c in 0..channels {
                let mut val = 0.0;
                for k in 0..kernel_size {
                    let pos = start + k;
                    if pos >= self.padding && pos < seq_len + self.padding {
                        let actual_pos = pos - self.padding;
                        val += input.data[(actual_pos, c)] * self.kernel.data[(k, c)];
                    }
                }
                output.data[(i, c)] = val;
            }
        }
        output
    }
}

/// 2D convolution layer (single channel for simplicity).
#[derive(Debug, Clone)]
pub struct Conv2D {
    pub kernel: Tensor, // (kernel_h, kernel_w)
    pub stride: usize,
    pub padding: usize,
}

impl Conv2D {
    pub fn new(kernel_h: usize, kernel_w: usize, stride: usize, padding: usize) -> Self {
        let scale = (2.0 / (kernel_h * kernel_w) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let vals: Vec<f64> = (0..kernel_h * kernel_w)
            .map(|_| rand::Rng::gen_range(&mut rng, -scale..scale))
            .collect();
        Conv2D {
            kernel: Tensor::new(kernel_h, kernel_w, vals),
            stride,
            padding,
        }
    }

    pub fn from_kernel(kernel: Tensor, stride: usize, padding: usize) -> Self {
        Conv2D { kernel, stride, padding }
    }

    /// Input/output treated as flat (h*w, 1). We store height/width externally.
    /// Returns (output_h, output_w) and the output tensor as flat (output_h*output_w, 1).
    pub fn forward(&self, input: &Tensor, in_h: usize, in_w: usize) -> (usize, usize, Tensor) {
        assert_eq!(input.rows(), in_h * in_w);
        let (kh, kw) = self.kernel.shape();
        let out_h = (in_h + 2 * self.padding - kh) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - kw) / self.stride + 1;

        let mut output = Tensor::zeros(out_h * out_w, 1);

        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut val = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let ih = oh * self.stride + ki;
                        let iw = ow * self.stride + kj;
                        if ih >= self.padding && ih < in_h + self.padding &&
                           iw >= self.padding && iw < in_w + self.padding {
                            let ai = ih - self.padding;
                            let aj = iw - self.padding;
                            val += input.data[(ai * in_w + aj, 0)] * self.kernel.data[(ki, kj)];
                        }
                    }
                }
                output.data[(oh * out_w + ow, 0)] = val;
            }
        }

        (out_h, out_w, output)
    }
}

/// Pooling type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolType {
    Max,
    Avg,
}

/// 2D pooling layer.
#[derive(Debug, Clone)]
pub struct Pool2D {
    pub pool_size: usize,
    pub stride: usize,
    pub pool_type: PoolType,
}

impl Pool2D {
    pub fn new(pool_size: usize, stride: usize, pool_type: PoolType) -> Self {
        Pool2D { pool_size, stride, pool_type }
    }

    pub fn forward(&self, input: &Tensor, in_h: usize, in_w: usize) -> (usize, usize, Tensor) {
        assert_eq!(input.rows(), in_h * in_w);
        let out_h = (in_h - self.pool_size) / self.stride + 1;
        let out_w = (in_w - self.pool_size) / self.stride + 1;
        let channels = input.cols();

        let mut output = Tensor::zeros(out_h * out_w, channels);

        for oh in 0..out_h {
            for ow in 0..out_w {
                for c in 0..channels {
                    let mut vals = Vec::new();
                    for pi in 0..self.pool_size {
                        for pj in 0..self.pool_size {
                            let ih = oh * self.stride + pi;
                            let iw = ow * self.stride + pj;
                            vals.push(input.data[(ih * in_w + iw, c)]);
                        }
                    }
                    output.data[(oh * out_w + ow, c)] = match self.pool_type {
                        PoolType::Max => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        PoolType::Avg => vals.iter().sum::<f64>() / vals.len() as f64,
                    };
                }
            }
        }

        (out_h, out_w, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_output_shape() {
        let conv = Conv1D::new(1, 3, 1, 0);
        let input = Tensor::new(10, 1, vec![1.0; 10]);
        let out = conv.forward(&input);
        assert_eq!(out.rows(), 8); // (10 - 3) / 1 + 1
        assert_eq!(out.cols(), 1);
    }

    #[test]
    fn test_conv1d_with_stride() {
        let conv = Conv1D::new(1, 3, 2, 0);
        let input = Tensor::new(10, 1, vec![1.0; 10]);
        let out = conv.forward(&input);
        assert_eq!(out.rows(), 4); // (10 - 3) / 2 + 1
    }

    #[test]
    fn test_conv1d_with_padding() {
        let conv = Conv1D::new(1, 3, 1, 1);
        let input = Tensor::new(10, 1, vec![1.0; 10]);
        let out = conv.forward(&input);
        assert_eq!(out.rows(), 10); // (10 + 2 - 3) / 1 + 1
    }

    #[test]
    fn test_conv1d_known_values() {
        let kernel = Tensor::new(3, 1, vec![1.0, 0.0, -1.0]);
        let conv = Conv1D { kernel, stride: 1, padding: 0 };
        let input = Tensor::new(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = conv.forward(&input);
        // [1*1 + 2*0 + 3*(-1), 2*1 + 3*0 + 4*(-1), 3*1 + 4*0 + 5*(-1)]
        assert_eq!(out.rows(), 3);
        assert!((out.data[(0, 0)] - (-2.0)).abs() < 1e-10);
        assert!((out.data[(1, 0)] - (-2.0)).abs() < 1e-10);
        assert!((out.data[(2, 0)] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2D::new(3, 3, 1, 0);
        let input = Tensor::new(16, 1, vec![1.0; 16]); // 4x4
        let (oh, ow, out) = conv.forward(&input, 4, 4);
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(out.rows(), 4);
    }

    #[test]
    fn test_conv2d_known_values() {
        let kernel = Tensor::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let conv = Conv2D::from_kernel(kernel, 1, 0);
        // 3x3 input
        let input = Tensor::new(9, 1, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let (oh, ow, out) = conv.forward(&input, 3, 3);
        assert_eq!((oh, ow), (2, 2));
        // (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
        assert!((out.data[(0, 0)] - 6.0).abs() < 1e-10);
        // (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
        assert!((out.data[(3, 0)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_maxpool2d_shape() {
        let pool = Pool2D::new(2, 2, PoolType::Max);
        let input = Tensor::new(16, 1, vec![1.0; 16]); // 4x4
        let (oh, ow, out) = pool.forward(&input, 4, 4);
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(out.rows(), 4);
    }

    #[test]
    fn test_avgpool2d_shape() {
        let pool = Pool2D::new(2, 2, PoolType::Avg);
        let input = Tensor::new(16, 1, vec![1.0; 16]); // 4x4
        let (oh, ow, out) = pool.forward(&input, 4, 4);
        assert_eq!((oh, ow), (2, 2));
    }

    #[test]
    fn test_maxpool2d_values() {
        let pool = Pool2D::new(2, 2, PoolType::Max);
        let input = Tensor::new(4, 1, vec![1.0, 3.0, 2.0, 4.0]); // 2x2
        let (oh, ow, out) = pool.forward(&input, 2, 2);
        assert_eq!((oh, ow), (1, 1));
        assert!((out.data[(0, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2D::new(3, 3, 1, 1);
        let input = Tensor::new(9, 1, vec![1.0; 9]); // 3x3
        let (oh, ow, _) = conv.forward(&input, 3, 3);
        assert_eq!((oh, ow), (3, 3)); // (3 + 2 - 3) / 1 + 1
    }
}
