use std::array;
use stmc_rs::marsaglia::Marsaglia;

pub struct IntegerPerceptron<const IDIM: usize, const ODIM: usize> {
    pub weights: [[i32; IDIM]; ODIM],
}

impl<const IDIM: usize, const ODIM: usize> Default for IntegerPerceptron<IDIM, ODIM> {
    fn default() -> Self {
        Self {
            weights: [[0; IDIM]; ODIM],
        }
    }
}

impl<const IDIM: usize, const ODIM: usize> IntegerPerceptron<IDIM, ODIM> {
    pub fn new(rng: &mut Marsaglia) -> Self {
        let norm = (2.0 / IDIM as f64).sqrt();
        const SCALE: f64 = 1.0;
        Self {
            weights: array::from_fn(|_| array::from_fn(|_| (rng.gauss() * norm * SCALE) as i32)),
        }
    }

    pub fn classify(&self, x: &[u8; IDIM]) -> usize {
        (0..ODIM)
            .max_by_key(|&i| {
                self.weights[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&wi, &xi)| wi as i64 * xi as i64)
                    .sum::<i64>()
            })
            .unwrap()
    }

    // For ODIM==1
    pub fn classify_binary(&self, x: &[u8; IDIM]) -> i64 {
        self.weights[0]
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi as i64 * xi as i64)
            .sum()
    }

    pub fn train(&mut self, data: &[[u8; IDIM]], labels: &[u8]) -> usize {
        let mut errors = 0;
        for (x, &target) in data.iter().zip(labels.iter()) {
            let predicted = self.classify(x);
            if predicted != target as usize {
                errors += 1;
                for j in 0..IDIM {
                    // Integer update: Add/Subtract the pixel value [0-255]
                    // This is equivalent to a learning rate of 1
                    self.weights[target as usize][j] += x[j] as i32;
                    self.weights[predicted][j] -= x[j] as i32;
                }
            }
        }
        errors
    }

    // For ODIM==1
    pub fn train_binary(&mut self, data: &[[u8; IDIM]], labels: &[bool]) -> usize {
        let mut errors = 0;
        for (x, &target) in data.iter().zip(labels.iter()) {
            let score = self.classify_binary(x);
            let predicted = score >= 0;
            if predicted != target {
                errors += 1;
                for j in 0..IDIM {
                    if target {
                        self.weights[0][j] += x[j] as i32;
                    } else {
                        self.weights[0][j] -= x[j] as i32;
                    }
                }
            }
        }
        errors
    }
}
