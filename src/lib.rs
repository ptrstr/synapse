pub mod utils;
pub mod neuron;
pub mod neural_network;
pub mod brain;

#[no_mangle]
pub extern fn neural_network_new(layers: Vec<usize>) -> neural_network::NeuralNetwork {
    neural_network::NeuralNetwork::new(layers)
}

#[no_mangle]
pub extern fn neural_network_train(neural_network: &mut neural_network::NeuralNetwork, inputs: Vec<f64>, outputs: Vec<f64>, learning_rate: f64) {
    neural_network.train(inputs, outputs, learning_rate)
}

#[no_mangle]
pub extern fn neural_network_guess(neural_network: neural_network::NeuralNetwork, inputs: Vec<f64>) -> Vec<f64> {
    neural_network.guess(inputs)
}

#[no_mangle]
pub extern fn brain_new(layers: Vec<usize>) -> brain::Brain {
    brain::Brain::new(layers)
}

#[no_mangle]
pub extern fn brain_mutate(brain: &mut brain::Brain) {
    brain.mutate()
}

#[no_mangle]
pub extern fn brain_guess(brain: brain::Brain, inputs: Vec<f64>) -> Vec<f64> {
    brain.guess(inputs)
}
