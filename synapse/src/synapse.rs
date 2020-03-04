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
		let mut weights = vec![];

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
						weights[i][j].push(rng.gen());
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
		let mut artifacts:Vec<Vec<f64>> = vec![vec![]];
		artifacts.push(inputs);
		for i in 0..self.neurons.len() - 1 {
			let mut new_artifacts:Vec<f64> = vec![];
			for j in 0..self.neurons[i].len() {
				let current_weights_start:usize = i * self.neurons[i+1].len();
				let current_weights_end:usize = current_weights_start + self.neurons[i+1].len();
				let current_weights:Vec<f64> = self.weights[i][current_weights_start..current_weights_end].to_vec();
				new_artifacts.push(self.neurons[i+1][j].run(artifacts[i].clone(), current_weights));
			}
			artifacts.push(new_artifacts);
		}
		return artifacts[artifacts.len() - 1].clone();
	}
}