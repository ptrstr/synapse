extern crate synapse;
extern crate rand;
use rand::Rng;

fn main() {
	let mut x = synapse::neural_network::NeuralNetwork::new(vec![2,2,1]);
	let learning_rate = 0.25;
	
	let data = [
		[
			vec![0.0, 0.0], vec![0.0]
		],
		[
			vec![0.0, 1.0], vec![1.0]
		],
		[
			vec![1.0, 0.0], vec![1.0]
		],
		[
			vec![0.0, 0.0], vec![0.0]
		]
	];

	let mut rng = rand::thread_rng();
	
	for _ in 0..20000 {
		let index = rng.gen_range(0, data.len());
		x.train(data[index][0].clone(), data[index][1].clone(), learning_rate);
	}
	let mut guess_inputs:Vec<f64> = vec![0.0,1.0];
	println!("{} XOR {} = {}", guess_inputs[0], guess_inputs[1], x.guess(guess_inputs.clone())[0]);
	guess_inputs = vec![0.0,0.0];
	println!("{} XOR {} = {}", guess_inputs[0], guess_inputs[1], x.guess(guess_inputs.clone())[0]);
}
