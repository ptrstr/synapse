use crate::neural_network;
extern crate rand;
use rand::Rng;

#[derive(Clone)]
pub struct Brain {
	neural_network: neural_network::NeuralNetwork,
	fitness: f64,
}

impl Brain {
	pub fn new(layers: Vec<usize>) -> Brain {
		Brain {
			neural_network: neural_network::NeuralNetwork::new(layers),
			fitness: 0.0_f64,
		}
	}

	pub fn guess(&self, inputs: Vec<f64>) -> Vec<f64> {
		self.neural_network.guess(inputs)
	}

	pub fn mutate(&mut self) {
		let mut mutated_neurons = self.neural_network.get_neurons();
		let mut mutated_weights = self.neural_network.get_weights();

		let mut rng = rand::thread_rng();

		for layer in 0..self.neural_network.get_neurons().len() {
			for neuron in 0..self.neural_network.get_neurons()[layer].len() {
				mutated_neurons[layer][neuron].add_bias(rng.gen_range(-1.0_f64, 1.0_f64));

				for weight in 0..self.neural_network.get_weights()[layer][neuron].len() {
					mutated_weights[layer][neuron][weight] += rng.gen_range(-1.0_f64, 1.0_f64);
				}
			}
		}

		self.neural_network.set_neurons(mutated_neurons);
		self.neural_network.set_weights(mutated_weights);
	}
}
