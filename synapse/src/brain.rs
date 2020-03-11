use crate::neural_network;

pub struct Brain {
	neural_network: neural_network::NeuralNetwork,
	fitness: f64,
}

impl Brain {
	pub fn new(layers: Vec<usize>) -> Brain {
		Brain {
			neural_network: neural_network::NeuralNetwork::new(layers),
		}
	}

	pub fn guess(&self, inputs: Vec<f64>) -> Vec<f64> {
		self.neural_network.guess(inputs)
	}

	pub fn mutate(&mut self) {

	}
}
