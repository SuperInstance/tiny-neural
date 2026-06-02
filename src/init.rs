use crate::Tensor;
use rand::Rng;
use rand_distr::{Normal, Distribution};

/// Xavier (Glorot) uniform initialization.
/// Draws from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_init(fan_in: usize, fan_out: usize) -> Tensor {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let mut rng = rand::thread_rng();
    let vals: Vec<f64> = (0..fan_in * fan_out)
        .map(|_| rng.gen_range(-limit..limit))
        .collect();
    Tensor::new(fan_in, fan_out, vals)
}

/// Xavier normal initialization.
/// Draws from N(0, stddev) where stddev = sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal_init(fan_in: usize, fan_out: usize) -> Tensor {
    let stddev = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let normal = Normal::new(0.0, stddev).unwrap();
    let mut rng = rand::thread_rng();
    let vals: Vec<f64> = (0..fan_in * fan_out)
        .map(|_| normal.sample(&mut rng))
        .collect();
    Tensor::new(fan_in, fan_out, vals)
}

/// He (Kaiming) initialization — for ReLU and variants.
/// Draws from N(0, stddev) where stddev = sqrt(2 / fan_in).
pub fn he_init(fan_in: usize, fan_out: usize) -> Tensor {
    let stddev = (2.0 / fan_in as f64).sqrt();
    let normal = Normal::new(0.0, stddev).unwrap();
    let mut rng = rand::thread_rng();
    let vals: Vec<f64> = (0..fan_in * fan_out)
        .map(|_| normal.sample(&mut rng))
        .collect();
    Tensor::new(fan_in, fan_out, vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_shape() {
        let t = xavier_init(100, 50);
        assert_eq!(t.shape(), (100, 50));
    }

    #[test]
    fn test_xavier_variance() {
        // With fan_in=1000, fan_out=1000, variance should be ~2/(1000+1000) = 0.001, stddev ~0.0316
        let t = xavier_init(1000, 1000);
        let vals = t.as_slice();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 0.02, "Mean should be near 0, got {}", mean);
        // Expected variance for uniform: limit^2/3 = (6/2000)^2 / 3 ≈ ... 
        // Actually limit = sqrt(6/2000) ≈ 0.0548, variance = limit^2/3 ≈ 0.001
        assert!((variance - 0.001).abs() < 0.001, "Variance should be ~0.001, got {}", variance);
    }

    #[test]
    fn test_xavier_normal_variance() {
        let t = xavier_normal_init(1000, 1000);
        let vals = t.as_slice();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let expected_var = 2.0 / 2000.0; // 0.001
        assert!(mean.abs() < 0.01, "Mean should be near 0");
        assert!((variance - expected_var).abs() < 0.001, "Variance should be ~0.001, got {}", variance);
    }

    #[test]
    fn test_he_init_shape() {
        let t = he_init(100, 50);
        assert_eq!(t.shape(), (100, 50));
    }

    #[test]
    fn test_he_init_variance() {
        // fan_in=1000, expected variance = 2/1000 = 0.002
        let t = he_init(1000, 1000);
        let vals = t.as_slice();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let expected_var = 2.0 / 1000.0;
        assert!(mean.abs() < 0.02, "Mean should be near 0");
        assert!((variance - expected_var).abs() < 0.002, "Variance should be ~0.002, got {}", variance);
    }
}
