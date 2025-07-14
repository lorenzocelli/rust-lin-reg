use std::{error::Error, fs::File, process, vec::Vec};

use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::error::LinalgError;

type Record = (i64, f64, f64, f64, f64);

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let file = File::open("Advertising.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut data: Vec<Record> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        data.push(record);
    }

    let n = data.len();

    let mut x: Array1<f64> = Array::zeros(n);
    let mut y: Array1<f64> = Array::zeros(n);

    // Extract 'TV' and 'sales' columns
    for i in 0..n {
        x[i] = data[i].1;
        y[i] = data[i].4;
    }

    let (b0, b1) = lin_reg(&x, &y);

    println!("[{}, {}]", b0, b1);

    // Compute the residual standard error
    let err = rse(&y, &(b0 + &x * b1));
    println!("RSE: {}", err);

    let mut x_mult: Array2<f64> = Array::ones((n, 4));

    // Extract 'TV', 'radio', and 'newspaper' columns
    for i in 0..n {
        x_mult[[i, 1]] = data[i].1;
        x_mult[[i, 2]] = data[i].2;
        x_mult[[i, 3]] = data[i].3;
    }

    let b = lin_reg_mult(&x_mult, &y)?;

    println!("{}", b);

    // Compute the average squared error for multiple regression
    let err_m = rse(&y, &(x_mult.dot(&b)));
    println!("RSE: {}", err_m);

    Ok(())
}

fn lin_reg(x: &Array1<f64>, y: &Array1<f64>) -> (f64, f64) {
    let x_mean: f64 = x.mean().unwrap();
    let y_mean: f64 = y.mean().unwrap();

    let x_dev = x - x_mean;
    let y_dev = y - y_mean;

    let b1_num = (&x_dev * &y_dev).sum();
    let b1_den = (&x_dev * &x_dev).sum();

    let b1: f64 = b1_num / b1_den;
    let b0: f64 = y_mean - b1 * x_mean;

    return (b0, b1);
}

fn lin_reg_mult(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let mut a = x.t().dot(x);
    let b: Array1<f64> = x.t().dot(y);

    // Avoid singular matrix
    for i in 0..x.shape()[1] {
        a[[i, i]] += 1e-10;
    }

    a.solvec(&b)
}

fn rse(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    let n = y.len() as f64;
    let err = y - y_hat;
    ((&err * &err).sum() / (n - 2.0)).sqrt()
}
