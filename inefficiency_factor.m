% Compute inefficiency factor of posterior draws

function inef = inefficiency_factor(x)

inef = 1./effective_sample_size_portion(x);
