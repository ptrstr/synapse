extern crate synapse;

fn main() {
/*
	let mut x = synapse::synapse::Synapse::new(vec![2,2,1]);
	for _ in 0..20000 {
		x.train(vec![0.0,0.0], vec![0.0], 0.5);
		x.train(vec![0.0,1.0], vec![1.0], 0.5);
		x.train(vec![1.0,0.0], vec![1.0], 0.5);
		x.train(vec![1.0,1.0], vec![0.0], 0.5);
	}
	let mut guess_inputs:Vec<f64> = vec![0.0,1.0];
	println!("{} XOR {} = {}", guess_inputs[0], guess_inputs[1], x.guess(guess_inputs.clone())[0]);
	guess_inputs = vec![0.0,0.0];
	println!("{} XOR {} = {}", guess_inputs[0], guess_inputs[1], x.guess(guess_inputs.clone())[0]);
*/
	let mut x = synapse::synapse::Synapse::new(vec![1,1,1]);
	let mut y = x.neurons[1][0].get_bias();
	x.neurons[1][0].add_bias(-1_f64 * y);
	x.neurons[1][0].add_bias(0.2);

	y = x.neurons[2][0].get_bias();
	x.neurons[2][0].add_bias(-1_f64 * y);
	x.neurons[2][0].add_bias(0.5);
	
	x.weights[0][0][0] = 0.6;
	x.weights[1][0][0] = 0.5;

	println!("Guess: {:?}", x.guess(vec![28_f64]));
}
