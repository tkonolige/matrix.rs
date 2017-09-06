#![feature(test)]
#![feature(step_trait)]
#![feature(specialization)]
#![feature(link_args)]
#![feature(trusted_len)]
#![feature(iterator_step_by)]

extern crate bit_vec;
extern crate itertools;
#[macro_use(s)]
extern crate ndarray;
extern crate num_integer;
extern crate num_traits;
extern crate rand;

extern crate test;

pub mod mat;

pub mod level1;
pub mod level2;
pub mod level3;

pub mod util;

#[cfg(test)]
mod tests;
