# Linear Regression in Rust

Codebase for the blog post ['Linear Regression in Rust'](http://lorenzocelli.me/2025/06/28/linear-regression-in-Rust.html).

This project implements a simple ordinary least squares linear regression model in Rust. It leverages [ndarray](https://github.com/rust-ndarray/ndarray) to store data points and [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) to solve the linear system.

The sample csv dataset comes from the online resources of the book ['Introduction to Statistical Learning'](https://www.statlearning.com/resources-python).

## Usage

Build the project with `cargo build` and run the exectuable (e.g. `./target/debug/rust-lin-reg`). The program will load the 'Advertising.csv' dataset, print the linear regression coefficients, and compute the residual standard error (RSE) for both simple and multiple regression.

```bash
[7.032593549127704, 0.04753664043301969]
RSE: 3.258656368650463
[2.938889369449337, 0.045764645455425176, 0.18853001691831814, -0.0010374930424140033]
RSE: 1.6769760888385674
```
