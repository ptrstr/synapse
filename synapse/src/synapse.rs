extern crate rand;
use rand::Rng;
use crate::neuron;

pub struct Synapse {
	neurons: Vec<Vec<neuron::Neuron>>,
	weights: Vec<Vec<Vec<f64>>>,
}

impl Synapse {
	pub fn new(layers: Vec<usize>) -> Synapse {
		let mut neurons = vec![];
		
		let mut rng = rand::thread_rng();
		let mut weights:Vec<Vec<Vec<f64>>> = vec![];

		for i in 0..layers.len() {
			// Initialize neurons
			let mut neurons_vector = vec![];
			for _ in 0..layers[i] {
				neurons_vector.push(neuron::Neuron::new());
			}
			neurons.push(neurons_vector);

			// Initialize Weights
			if i < layers.len() - 1 {
				weights.push(vec![]);
				for j in 0..layers[i] {
					weights[i].push(vec![]);
					for _ in 0..layers[i+1] {
						weights[i][j].push(rng.gen_range(-1_f64, 1_f64));
					}
				}
			}
		}

		Synapse {
			neurons: neurons,
			weights: weights,
		}
	}

	pub fn guess(&self, inputs: Vec<f64>) -> Vec<f64> {
		let mut artifacts:Vec<Vec<f64>> = vec![];
		artifacts.push(inputs);
		for i in 0..self.neurons.len() - 1 {
			let mut new_artifacts:Vec<f64> = vec![];
			for j in 0..self.neurons[i].len() {
				new_artifacts.push(self.neurons[i+1][j].run(artifacts[i].clone(), self.weights[i][j].clone()));
			}
			artifacts.push(new_artifacts);
		}
		println!("{:?}", artifacts);
		return artifacts[artifacts.len() - 1].clone();
	}

	pub fn train(&mut self, inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>, iterations: usize, learning_rate: f64) {
		let mut rng = rand::thread_rng();
		for _ in 0..iterations {
			let current_io = rng.gen_range(0, inputs.len());
			let guess_outputs = self.guess(inputs[current_io].clone());
			let mut error_outputs:Vec<f64> = vec![];
			for i in 0..outputs[current_io].len() {
				error_outputs.push(outputs[current_io][i] - guess_outputs[i]);
			}

			let mut errors:Vec<Vec<f64>> = vec![];
			for i in self.neurons.len() - 1..=0 {
				
			}
			for i in 0..error_outputs.len() {
				println!("{}: [Guess: {}], [Target: {}], [Error: {}]", i, guess_outputs[i], outputs[current_io][i], error_outputs[i]);
			}
		}
	}
}
