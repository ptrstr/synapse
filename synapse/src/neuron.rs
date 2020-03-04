extern crate rand;
use rand::Rng;

pub struct Neuron {
	bias: f64,
}

impl Neuron {
	pub fn new() -> Neuron {
		let mut rng = rand::thread_rng();

		Neuron {
			bias: rng.gen(),
		}
	}

	pub fn get_bias(&self) -> f64 {
		return self.bias;
	}

	pub fn add_bias(&mut self, amount: f64) {
		self.bias += amount;
	}

	pub fn run(&self, inputs: Vec<f64>, weights: Vec<f64>) -> f64 {
		if inputs.len() != weights.len() {
			return 0.0;
		}

		let mut sum:f64 = self.bias;
		for i in 0..inputs.len() {
			sum += inputs[i] * weights[i];
		}

		return sum;
	}
}
