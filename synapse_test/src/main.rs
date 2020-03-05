extern crate synapse;

fn main() {
	let mut x = synapse::synapse::Synapse::new(vec![2,2,2]);
	x.train(vec![vec![4.0,6.0]], vec![vec![3.0, 5.0]], 1,0.5);
}
