# Perceptron

Exploring MNIST digit classification with perceptron models.

For comparison with other MNIST models see, see [FF](https://github.com/jesper-olsen/forward-forward-rs) and [Engram](https://github.com/jesper-olsen/engram).

## Getting Started

Clone the repository:

```sh
git clone https://github.com/jesper-olsen/perceptron.git
cd perceptron 
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the models:

```
cargo run --release 

Model 0/1 - Epoch 0: errors = 9192 = 15.320%
Model 0/1 - Epoch 1: errors = 7993 = 13.322%
Model 0/1 - Epoch 2: errors = 7554 = 12.590%
Model 0/1 - Epoch 3: errors = 7360 = 12.267%
Model 0/1 - Epoch 4: errors = 7277 = 12.128%
Model 0/1 - Epoch 5: errors = 7188 = 11.980%
Model 0/1 - Epoch 6: errors = 7112 = 11.853%
Model 0/1 - Epoch 7: errors = 7098 = 11.830%
Model 0/1 - Epoch 8: errors = 6921 = 11.535%
Model 0/1 - Epoch 9: errors = 7055 = 11.758%
Ensemble Accuracy: 8985/10000 = 89.85%
```

```
cargo run --release -- --ensemble-size 9

Model 0/9 - Epoch 0: errors = 9192 = 15.320%
Model 0/9 - Epoch 1: errors = 7993 = 13.322%
..snip..
Ensemble Accuracy: 9198/10000 = 91.98%
```
## References

* [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
