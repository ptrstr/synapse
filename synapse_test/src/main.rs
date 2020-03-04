extern crate synapse;

fn main() {
	let x = synapse::synapse::Synapse::new(vec![2,2,2]);
	x.guess(vec![4.0,6.0]);
}
