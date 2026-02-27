//use ferro_cell::perceptron::Perceptron;
use ferro_cell::integer_perceptron::IntegerPerceptron;
use mnist::{Mnist, error::MnistError};
use stmc_rs::marsaglia::Marsaglia;

//fn calc_stat(model: &Perceptron<784, 10>) {
//    for (i, row) in model.weights.iter().enumerate() {
//        let max = row.iter().map(|&w| w.abs()).max().unwrap();
//        let avg = row.iter().map(|&w| w.abs() as f64).sum::<f64>() / row.len() as f64;
//        println!("  Weights: Class {i}: max_abs={max:>6}, avg_abs={avg:>8.2}");
//    }
//}

fn main() -> Result<(), MnistError> {
    let mut rng = Marsaglia::new(42, 0, 0, 0);
    let data = Mnist::load("MNIST")?;

    // collect pixels into owned arrays
    //let mut train_images: Vec<[f64; 784]> = data.train_images.iter().map(|img| img.as_f64_array()).collect();
    //let test_images: Vec<[f64; 784]> = data.test_images.iter().map(|img| img.as_f64_array()).collect();
    let mut train_images: Vec<[u8; 784]> = data.train_images.iter().map(|img| img.pixels).collect();
    let test_images: Vec<[u8; 784]> = data.test_images.iter().map(|img| img.pixels).collect();
    let mut train_labels: Vec<u8> = data.train_labels.clone();
    //let mut model = Perceptron::<784, 10>::new(&mut rng);
    let mut model = IntegerPerceptron::<784, 10>::new(&mut rng);

    const MAX_EPOCHS: usize = 10;
    for epoch in 0..MAX_EPOCHS {
        // Shuffle training samples
        let n = train_images.len();
        for i in (1..n).rev() {
            let j = (rng.uni() * (i + 1) as f64) as usize;
            train_images.swap(i, j);
            train_labels.swap(i, j);
        }

        //let lr = 1.0;
        //let err = model.train(&train_images, &train_labels, lr);
        let errors = model.train(&train_images, &train_labels);
        let total = train_images.len();
        let rate = (100 * errors) as f64 / total as f64;
        println!("Epoch {epoch}: errors = {errors}/{total} =  = {rate:.3}%");

        //calc_stat(&model);
    }

    let correct = test_images
        .iter()
        .zip(data.test_labels.iter())
        .filter(|(image, label)| model.classify(image) == **label as usize)
        .count();
    let total = data.test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;

    println!("Accuracy: {correct}/{total} = {acc:.2}%");

    Ok(())
}
