use clap::Parser;
use mnist::{Mnist, error::MnistError};
use perceptron::integer_perceptron::IntegerPerceptron;
use stmc_rs::marsaglia::Marsaglia;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 1)]
    /// Number of classifiers in ensemble
    pub ensemble_size: usize,
}

const MAX_EPOCHS: usize = 10;

// Fisher-Yates shuffling
fn shuffle_together<A, B>(a: &mut Vec<A>, b: &mut Vec<B>, rng: &mut Marsaglia) {
    let n = a.len();
    for i in (1..n).rev() {
        let j = (rng.uni() * (i + 1) as f64) as usize;
        a.swap(i, j);
        b.swap(i, j);
    }
}

fn train_many(ensemble_size: usize) -> Result<(), MnistError> {
    let mut rng = Marsaglia::new(42, 0, 0, 0);
    let data = Mnist::load("MNIST")?;

    // collect pixels into owned arrays
    let mut train_images: Vec<[u8; 784]> = data.train_images.iter().map(|img| img.pixels).collect();
    let test_images: Vec<[u8; 784]> = data.test_images.iter().map(|img| img.pixels).collect();
    let mut train_labels: Vec<u8> = data.train_labels.clone();

    let mut models: Vec<IntegerPerceptron<784, 10>> = (0..ensemble_size)
        .map(|_| IntegerPerceptron::new(&mut rng))
        .collect();

    for (i, model) in models.iter_mut().enumerate() {
        for epoch in 0..MAX_EPOCHS {
            shuffle_together(&mut train_images, &mut train_labels, &mut rng);
            let errors = model.train(&train_images, &train_labels);
            let rate = (100 * errors) as f64 / train_images.len() as f64;
            println!("Model {i}/{ensemble_size} - Epoch {epoch}: errors = {errors} = {rate:.3}%");
        }
    }

    // Evaluate with majority voting
    let correct = test_images
        .iter()
        .zip(data.test_labels.iter())
        .filter(|(image, label)| {
            let mut votes = [0usize; 10];
            for model in &models {
                votes[model.classify(image)] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, v)| v)
                .map(|(i, _)| i)
                .unwrap()
                == **label as usize
        })
        .count();

    let total = test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;
    println!("Ensemble Accuracy: {correct}/{total} = {acc:.2}%");

    Ok(())
}
fn main() -> Result<(), MnistError> {
    let args = Args::parse();
    train_many(args.ensemble_size)
}
