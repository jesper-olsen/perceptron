use std::array;

pub struct IntegerPerceptron<const IDIM: usize, const ODIM: usize> {
    // Using i32 for weights to allow for growth during training
    pub weights: [[i32; IDIM]; ODIM],
}

impl<const IDIM: usize, const ODIM: usize> IntegerPerceptron<IDIM, ODIM> {
    pub fn new() -> Self {
        Self {
            weights: [[0; IDIM]; ODIM],
        }
    }

    pub fn output(&self, x: &[u8; IDIM]) -> [i64; ODIM] {
        array::from_fn(|i| {
            self.weights[i]
                .iter()
                .zip(x.iter())
                // Use i64 for the sum to prevent overflow during dot product
                .map(|(&wi, &xi)| wi as i64 * xi as i64)
                .sum::<i64>()
        })
    }

    pub fn classify(&self, x: &[u8; IDIM]) -> usize {
        let scores = self.output(x);
        scores
            .iter()
            .enumerate()
            .max_by_key(|&(_, score)| score)
            .map(|(index, _)| index)
            .unwrap()
    }

    pub fn train(&mut self, data: &[[u8; IDIM]], labels: &[u8]) -> f64 {
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
        errors as f64 / data.len() as f64
    }
}
