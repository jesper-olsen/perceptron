use engram::slide_3x3_window;
use mnist::{Mnist, error::MnistError};
use std::array;

fn print_patch(key: u16) {
    let pattern: [_; 9] = array::from_fn(|i| if (key >> i) & 1 == 1 { 'X' } else { '.' });
    println!("Pattern {key}:");
    println!("{}{}{}", pattern[0], pattern[1], pattern[2]);
    println!("{}{}{}", pattern[3], pattern[4], pattern[5]);
    println!("{}{}{}", pattern[6], pattern[7], pattern[8]);
}

fn main() -> Result<(), MnistError> {
    let mut cfg = [[0u64; 512]; 10];
    let data = Mnist::load("MNIST")?;
    for (image, label) in data.train_images.iter().zip(data.train_labels.iter()) {
        for patch in slide_3x3_window(image) {
            //println!("patch at {},{}", patch.x, patch.y);
            let mut u: u16 = 0;
            for i in 0..9 {
                if patch.pixels[i] > 25 {
                    u |= 1 << i
                }
            }
            cfg[*label as usize][u as usize] += 1;
        }
    }

    for i in 0..512 {
        //let has_center_ink = (i >> 4) & 1 == 1;
        print_patch(i as u16);
        for j in 0..10 {
            print!("D{j}: {:6} ", cfg[j][i]);
        }
        println!("\n-----------------");
    }

    Ok(())
}
