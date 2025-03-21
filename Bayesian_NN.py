import numpy as np
import pymc as pm

floatX = np.float64
rng = np.random.default_rng()

def construct_minibatch_nn(X_train, Y_train, hidden_layers=[5, 5], batch_size=50, sigma = 1, mu = 0):

    # Initialize random weights for each layer
    rng.normal()
    init_vals = [rng.normal(loc = mu, scale = sigma, size=(X_train.shape[1], hidden_layers[0])).astype(floatX)]
    for i in range(len(hidden_layers) - 1):
        init_vals.append(rng.normal(loc = mu, scale = sigma, size=(hidden_layers[i], hidden_layers[i + 1])).astype(floatX))
    init_out = rng.normal(loc = mu, scale = sigma, size=hidden_layers[-1]).astype(floatX)
    
    coords = {
        "train_cols": np.arange(X_train.shape[1]),
        "obs_id": np.arange(X_train.shape[0]),
    }
    for i, n_hidden in enumerate(hidden_layers):
        coords[f"hidden_layer_{i+1}"] = np.arange(n_hidden)
    
    with pm.Model(coords=coords) as neural_network:
        X_data = pm.Data("X_data", X_train, dims=("obs_id", "train_cols"))
        Y_data = pm.Data("Y_data", Y_train, dims="obs_id")
        
        ann_input, ann_output = pm.Minibatch(X_data, Y_data, batch_size=batch_size)
        
        # Initialize weights
        weights = [pm.Normal(f"w_0", mu, sigma=sigma, initval=init_vals[0], dims=("train_cols", "hidden_layer_1"))]
        for i in range(1, len(hidden_layers)):
            weights.append(pm.Normal(f"w_{i}", mu, sigma=sigma, initval=init_vals[i], dims=(f"hidden_layer_{i}", f"hidden_layer_{i+1}")))
        
        weights_out = pm.Normal("w_out", mu, sigma=sigma, initval=init_out, dims=f"hidden_layer_{len(hidden_layers)}")
        
        # Build network
        activation = pm.math.tanh(pm.math.dot(ann_input, weights[0]))
        for i in range(1, len(weights)):
            activation = pm.math.tanh(pm.math.dot(activation, weights[i]))
        act_out = pm.math.sigmoid(pm.math.dot(activation, weights_out))
        
        out = pm.Bernoulli("out", act_out, observed=ann_output, total_size=X_train.shape[0])
    
    return neural_network

def construct_nn(X_train, Y_train, hidden_layers=[5, 5], sigma = 1, mu = 0):

    # Initialize random weights for each layer
    rng = np.random.default_rng()  # Ensure RNG is initialized
    floatX = np.float64  # Define floating point precision

    init_vals = [rng.normal(loc = mu, scale = sigma, size=(X_train.shape[1], hidden_layers[0])).astype(floatX)]
    for i in range(len(hidden_layers) - 1):
        init_vals.append(rng.normal(loc = mu, scale = sigma, size=(hidden_layers[i], hidden_layers[i + 1])).astype(floatX))
    init_out = rng.normal(loc = mu, scale = sigma, size=hidden_layers[-1]).astype(floatX)
    
    coords = {
        "train_cols": np.arange(X_train.shape[1]),
        "obs_id": np.arange(X_train.shape[0]),
    }
    for i, n_hidden in enumerate(hidden_layers):
        coords[f"hidden_layer_{i+1}"] = np.arange(n_hidden)
    
    with pm.Model(coords=coords) as neural_network:
        # Use full dataset directly
        X_data = pm.Data("X_data", X_train, dims=("obs_id", "train_cols"))
        Y_data = pm.Data("Y_data", Y_train, dims="obs_id")

        # Initialize weights
        weights = [pm.Normal(f"w_0", mu, sigma=sigma, initval=init_vals[0], dims=("train_cols", "hidden_layer_1"))]
        for i in range(1, len(hidden_layers)):
            weights.append(pm.Normal(f"w_{i}", mu, sigma=sigma, initval=init_vals[i], dims=(f"hidden_layer_{i}", f"hidden_layer_{i+1}")))

        weights_out = pm.Normal("w_out", mu, sigma=sigma, initval=init_out, dims=f"hidden_layer_{len(hidden_layers)}")

        # Build network (full batch training)
        activation = pm.math.tanh(pm.math.dot(X_data, weights[0]))
        for i in range(1, len(weights)):
            activation = pm.math.tanh(pm.math.dot(activation, weights[i]))
        act_out = pm.math.sigmoid(pm.math.dot(activation, weights_out))

        # Bernoulli likelihood (without minibatches)
        out = pm.Bernoulli("out", act_out, observed=Y_data)  # Removed `total_size`

    return neural_network

def sample_posterior_predictive(X_test, Y_test, trace, hidden_layers=[5, 5]):
    coords = {
        "train_cols": np.arange(X_test.shape[1]),
        "obs_id": np.arange(X_test.shape[0]),
    }
    for i, n_hidden in enumerate(hidden_layers):
        coords[f"hidden_layer_{i+1}"] = np.arange(n_hidden)
    
    with pm.Model(coords=coords):
        ann_input = X_test
        ann_output = Y_test
        
        weights = [pm.Flat(f"w_0", dims=("train_cols", "hidden_layer_1"))]
        for i in range(1, len(hidden_layers)):
            weights.append(pm.Flat(f"w_{i}", dims=(f"hidden_layer_{i}", f"hidden_layer_{i+1}")))
        
        weights_out = pm.Flat("w_out", dims=f"hidden_layer_{len(hidden_layers)}")
        
        activation = pm.math.tanh(pm.math.dot(ann_input, weights[0]))
        for i in range(1, len(weights)):
            activation = pm.math.tanh(pm.math.dot(activation, weights[i]))
        act_out = pm.math.sigmoid(pm.math.dot(activation, weights_out))
        
        out = pm.Bernoulli("out", act_out, observed=ann_output)
        return pm.sample_posterior_predictive(trace)