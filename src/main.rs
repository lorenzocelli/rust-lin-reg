use std::{
    env,
    vec::Vec,
    error::Error, 
    ffi::OsString,
    fs::File,
    process,
};

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
    let file = File::open(get_arg(1)?)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        x.push(record.1);
        y.push(record.4);
    }

    let (b0, b1) = lin_reg(&x, &y);

    println!("[{}, {}]", b0, b1);

    let n = x.len();
    let mut x_arr = Array::<f64, _>::ones((n, 2).f());
    let y_arr = Array::from_shape_vec((n).f(), y).unwrap();

    for i in 0..n {
        x_arr[[i, 1]] = x[i];
    }

    let b = lin_reg_mult(&x_arr, &y_arr)?;

    println!("{}", b);

    Ok(())
}

fn lin_reg (x: &Vec<f64>, y: &Vec<f64>) -> (f64, f64) {
    let mut x_mean: f64 = 0.0;
    let mut y_mean: f64 = 0.0;

    for i in 0..x.len() {
        x_mean += x[i];
        y_mean += y[i];
    }

    x_mean /= x.len() as f64;
    y_mean /= y.len() as f64;

    let mut b1_num: f64 = 0.0;
    let mut b1_den: f64 = 0.0;

    for i in 0..x.len() {
        let x_dev: f64 = x[i] - x_mean;
        b1_num += x_dev * (y[i] - y_mean);
        b1_den += x_dev * x_dev;
    }

    let b1: f64 = b1_num / b1_den;
    let b0: f64 = y_mean - b1 * x_mean;

    return (b0, b1);
}

fn lin_reg_mult (x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let mut a =  x.t().dot(x);
    let b : Array1<f64> = x.t().dot(y);
    
    // Avoid singular matrix
    for i in 0..x.shape()[1] {
        a[[i, i]] += 1e-8; 
    }

    a.solvec(&b)
}

fn get_arg(i: usize) -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(i) {
        None => Err(From::from("expected 1 argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}
