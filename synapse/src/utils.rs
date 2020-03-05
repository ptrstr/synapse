pub fn sigmoid(value: f64) -> f64 {
	1_f64 / (1_f64 + f64::powf(std::f64::consts::E, -value))
}

pub fn dsigmoid(value: f64) -> f64 {
	sigmoid(value) * (1_f64 - sigmoid(value))
}
