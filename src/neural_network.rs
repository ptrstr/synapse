extern crate rand;
use rand::Rng;

// extern crate json;
// use json::object;
//
// use std::fs;
// use std::path;
// use std::io::prelude::*;
// use std::error::Error;

use crate::{ neuron, utils };

#[derive(Clone)]
pub struct NeuralNetwork {
	neurons: Vec<Vec<neuron::Neuron>>,
	weights: Vec<Vec<Vec<f64>>>,
}

impl NeuralNetwork {
	pub fn new(layers: Vec<usize>) -> NeuralNetwork {
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


		NeuralNetwork {
			neurons: neurons,
			weights: weights,
		}
	}

	pub fn get_neurons(&self) -> Vec<Vec<neuron::Neuron>> {
		return self.neurons.clone();
	}

	pub fn set_neurons(&mut self, neurons: Vec<Vec<neuron::Neuron>>) {
		self.neurons = neurons;
	}

	pub fn get_weights(&self) -> Vec<Vec<Vec<f64>>> {
		return self.weights.clone();
	}

	pub fn set_weights(&mut self, weights: Vec<Vec<Vec<f64>>>) {
		self.weights = weights;
	}

	fn feed_forward(&self, inputs: Vec<f64>) -> Vec<Vec<f64>> {
		let mut artifacts:Vec<Vec<f64>> = vec![];
		artifacts.push(inputs);
		for i in 0..self.neurons.len() - 1 {
			let mut new_artifacts:Vec<f64> = vec![];
			for j in 0..self.neurons[i+1].len() {
				let mut current_weights:Vec<f64> = vec![];
				for k in 0..self.neurons[i].len() {
					current_weights.push(self.weights[i][k][j]);
				}
				new_artifacts.push(self.neurons[i+1][j].run(artifacts[i].clone(), current_weights));
			}
			artifacts.push(new_artifacts);
		}
		return artifacts;
	}

	pub fn guess(&self, inputs: Vec<f64>) -> Vec<f64> {
		let guesses = self.feed_forward(inputs);
		return guesses[guesses.len() - 1].clone();
	}

	pub fn train(&mut self, inputs: Vec<f64>, outputs: Vec<f64>, learning_rate: f64) {
		let guesses:Vec<Vec<f64>> = self.feed_forward(inputs);

		// Errors are in reverse order
		let mut errors:Vec<Vec<f64>> = vec![vec![]];
		for i in 0..outputs.len() {
			errors[0].push(outputs[i] - guesses[guesses.len() - 1][i]);
		}

		for cur_layer in (1..self.neurons.len()).rev() {
			let mut gradients:Vec<f64> = guesses[cur_layer].clone();
			let mut weight_deltas:Vec<Vec<f64>> = vec![];
			for cur_neuron in 0..self.neurons[cur_layer].len() {
				gradients[cur_neuron] = utils::dsigmoid(gradients[cur_neuron]) * errors[errors.len() - 1][cur_neuron] * learning_rate;
				self.neurons[cur_layer][cur_neuron].add_bias(gradients[cur_neuron]);
			}

			errors.push(vec![0.0; self.neurons[cur_layer-1].len()]);
			for prev_neuron in 0..self.neurons[cur_layer-1].len() {
				weight_deltas.push(vec![]);
				for cur_neuron in 0..self.neurons[cur_layer].len() {
					weight_deltas[prev_neuron].push(gradients[cur_neuron] * guesses[cur_layer - 1][prev_neuron]);
				}

				for prev_weight in 0..self.weights[cur_layer-1][prev_neuron].len() {
					self.weights[cur_layer-1][prev_neuron][prev_weight] += weight_deltas[prev_neuron][prev_weight];
					let new_error = errors[errors.len() - 2][prev_weight] * self.weights[cur_layer-1][prev_neuron][prev_weight];
					let cur_error_layer = errors.len() - 1;
					errors[cur_error_layer][prev_neuron] += new_error;
				}
			}
		}
	}

	// TODO: Add save/load
	/*pub fn save(&self, path: &str) {
		// Setup file before anything
		let mut extension_path = String::from(path);
		if !extension_path.ends_with(".json") {
			extension_path.push_str(".json");
		}

		let full_path = path::Path::new(&extension_path);
		let full_path_display = full_path.display();

		let mut file = match fs::File::create(&full_path) {
			Err(reason) => panic!("Error: Could not create {}! Reason: {}", full_path_display, reason.description()),
			Ok(file) => file,
		};

		let neuron_values: Vec<Vec<f64>> = self.neurons.iter().map(|i| i.iter().map(|j| j.get_bias()).collect()).collect();

		let mut data = object!{
			"neurons" => neuron_values,
			"weights" => self.weights.clone(),
		};

		let json_data = json::stringify(data);

		match file.write_all(json_data.as_bytes()) {
			Err(reason) => panic!("Error: Could not write to {}! Reason: {}", full_path_display, reason.description()),
			Ok(_) => println!("successfully wrote to {}", full_path_display),
		}
	}

	pub fn load(path: &str) -> NeuralNetwork {
		// Open file before anything else
		let full_path = path::Path::new(&path);
		let full_path_display = full_path.display();

		let mut file = match fs::File::open(&full_path) {
			Err(reason) => panic!("Error: Could not open {}! Reason: {}", full_path_display, reason.description()),
			Ok(file) => file,
		};

		let mut json_data = String::new();
    	match file.read_to_string(&mut json_data) {
        	Err(reason) => panic!("Error: Could not read {}: {}", full_path_display, reason.description()),
        	Ok(_) => print!("{} contains:\n{}", full_path_display, json_data),
    	};

		println!("{:?}", json::parse(&json_data));

		let mut neurons = vec![];
		let mut weights:Vec<Vec<Vec<f64>>> = vec![];



		NeuralNetwork {
			neurons: neurons,
			weights: weights,
		}
	}*/
}
