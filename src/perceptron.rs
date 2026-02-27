use std::array;
use std::fmt;
use stmc_rs::marsaglia::Marsaglia;

pub struct Perceptron<const IDIM: usize, const ODIM: usize> {
    // weights[output_index][input_index]
    pub weights: [[f64; IDIM]; ODIM],
}

impl<const IDIM: usize, const ODIM: usize> Default for Perceptron<IDIM, ODIM> {
    fn default() -> Self {
        Self {
            weights: [[0.0; IDIM]; ODIM],
        }
    }
}

impl<const IDIM: usize, const ODIM: usize> Perceptron<IDIM, ODIM> {
    pub fn new(rng: &mut Marsaglia) -> Self {
        let norm = (2.0 / IDIM as f64).sqrt();
        Self {
            weights: array::from_fn(|_| array::from_fn(|_| rng.gauss() * norm)),
        }
    }

    /// Calculates the raw activation for all output classes
    pub fn output(&self, x: &[f64; IDIM]) -> [f64; ODIM] {
        array::from_fn(|i| {
            self.weights[i]
                .iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>()
        })
    }

    /// Returns the index (digit) with the highest activation
    pub fn classify(&self, x: &[f64; IDIM]) -> usize {
        let scores = self.output(x);
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    /// Multi-class Perceptron Learning Rule
    pub fn train(&mut self, data: &[[f64; IDIM]], labels: &[u8], lr: f64) -> f64 {
        assert_eq!(data.len(), labels.len());
        let mut errors = 0.0;

        for (x, &target) in data.iter().zip(labels.iter()) {
            let predicted = self.classify(x);

            if predicted != target as usize {
                errors += 1.0;

                // Perceptron Update Rule:
                // Increase weights for the correct class
                // Decrease weights for the incorrectly predicted class
                for j in 0..IDIM {
                    self.weights[target as usize][j] += lr * x[j];
                    self.weights[predicted][j] -= lr * x[j];
                }
            }
        }
        errors / data.len() as f64
    }
}

impl<const IDIM: usize, const ODIM: usize> fmt::Display for Perceptron<IDIM, ODIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Multi-Class Perceptron ({} -> {})", IDIM, ODIM)?;
        for (i, row) in self.weights.iter().enumerate() {
            write!(f, "Digit {}: ", i)?;
            for w in row.iter().take(5) {
                // Print first 5 weights for brevity
                write!(f, "{w:>8.2} ")?;
            }
            writeln!(f, "...")?;
        }
        Ok(())
    }
}
