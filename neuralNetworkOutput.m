function neuralNetworkOutput
	%training_sample = mackeyGlassOut();
	
	training_sample = load('consumption.txt');
	
	training_sample = getNormalizedValues(training_sample(:,1));
		
	number_of_rows = size(training_sample,1);
	number_of_inputs = 3;
		
	number_of_membership_funcs = 3;
	
	c = zeros(number_of_inputs,number_of_membership_funcs);
	
	c(1,:) = 0;
	c(2,:) = 0.5;
	c(3,:) = 1;
		
	% main neural network learning start
	
	start_index = 15;
	number_of_training_samples = 1650;
	number_of_values_to_predict = 10;
	number_of_values_to_predict = 10;
	
	[weights,system_output,t_sample] = trainNetwork(number_of_training_samples,
													training_sample,
													c,
													start_index);
													
	clf();										
	
	true_value = training_sample(start_index : start_index + number_of_training_samples - 1);
	
	%save result.m true_value system_output
		
	%plot(true_value,'r');
	%hold on;
	%plot(system_output(2 : end),'b');
		
	offset = 100;
	
	history_length = 14;
		
	memory = training_sample(start_index + offset : start_index + number_of_values_to_predict + history_length + offset - 1);
	
	prediction = getPrediction(number_of_values_to_predict,
									memory,
									weights,
									c,
									start_index);
											
	plot(memory(history_length : end), 'r');					
	hold on;
	plot(prediction,'b');
	
	error = sum((prediction - memory(history_length : end-1)) .^ 2)/2
	
	
		
end

function [weights, system_output,t_sample] = trainNetwork(number_of_learning_signals,training_sample,c,start_index)
	number_of_inputs = size(c,1);
	
	system_output = zeros(number_of_learning_signals,1);
	mu = zeros(size(c));
	
	c_index = 1;
	sigma = 0.5;
	
	t_sample = size(system_output,1);
	
	error = size(system_output,1);
	
	weights = zeros(number_of_inputs,1);
	
	end_value = number_of_learning_signals + start_index;
	
	while(start_index < end_value)
		
		learning_signal = training_sample(start_index);
		
		input_sample = getInputSampleWithIndex(training_sample, start_index);
	
		mu = getMembershipValues(input_sample,c,sigma);
		
		aggregated_values = getAggregatedValue(mu);
		
		weighted_values = aggregated_values .* weights;
		
		sum_aggregated_values = sum(aggregated_values);
		
		sum_weighted_values = sum(weighted_values);
		
		phi = (aggregated_values / sum_aggregated_values);
				
		res = weights' * phi;		
		
		error(c_index) = sum((learning_signal - res)^2) / 2;
		
		weights = calculateNewWeights(phi, learning_signal);
		
		c = calculate_new_centers(input_sample, c_index, c);
		
		start_index = start_index + 1;
		
		t_sample(c_index) = learning_signal;
		
		system_output(c_index) = res;
		c_index = c_index + 1;
		
		
	end;
	
end;

function prediction_result = getPrediction(number_of_values_to_predict, memory, weights, centers,start_index)
	prediction_result = zeros(number_of_values_to_predict,1);
	counter = 1;
	sigma = 0.5;
		
	while(counter <= number_of_values_to_predict)
		learning_signal = memory(start_index);
		
		input_sample = getInputSampleWithIndex(memory, start_index);
	
		mu = getMembershipValues(input_sample,centers,sigma);
		
		aggregated_values = getAggregatedValue(mu);
		
		weighted_values = aggregated_values .* weights;
		
		sum_aggregated_values = sum(aggregated_values);
		
		sum_weighted_values = sum(weighted_values);
		
		phi = (aggregated_values / sum_aggregated_values);
				
		res = weights' * phi;
				
		memory(end + 1) = res;
		
		weights = calculateNewWeights(phi, learning_signal);
		centers = calculate_new_centers(input_sample, counter, centers);
		
		prediction_result(counter) = res;
		
		counter = counter + 1;
		start_index = start_index + 1;
	end;
end;

function new_centers = calculate_new_centers(x, index,old_centers)
	new_centers = zeros(size(old_centers));
	step = 0.1;
	radius = 0.33;
	
	if(index == 1)
		for i = 1 : size(new_centers)
			new_centers(i,:) = x;
		end;
	else
		distances = get_distance(x, old_centers);
		
		for i = 1: size(distances)
			if(distances(i) <= radius)
				
				for k = 1: size(new_centers)
					new_centers(k,:) = (old_centers(k,:) + x') / 2;
				end;
				
			elseif distances(i) > radius && distances(i) <= 2 * radius
			
				neighbour = Epanechnikov_neighbour(x, old_centers, radius);
				
				for i = 1: size(new_centers)
					new_centers(i,:) = old_centers(i,:) + step * neighbour * (x - old_centers(i,:)')';
				end;
				
			end;
		end;
	end;
	
end;

function distances = get_distance(x,y)
	distances = zeros(size(x,1), 1);
	
	for i = 1 : size(x, 1)
		distances(i) = sqrt(sum( (x - y(i,:)') .^ 2));
	end;
end;

function result = Epanechnikov_neighbour(x,center,radius)	
	distances = get_distance(x, center);
		
	result = max(0.1 - ((distances/ (2*radius)) .^ 2) );
end;

function normalized = getNormalizedValues(values)
	normalized = values;
	for i = 1: size(values)
		normalized(i) = (values(i) - min(values))/(max(values) - min(values));
	end;
end;

function new_weights = calculateNewWeights(phi, learning_signal)
	new_weights = pinv(sum(phi * phi')) * sum(phi * learning_signal);
end;


function aggregatedValues = getAggregatedValue(membership_values)
	aggregatedValue = 1;
	
	aggregatedValues = zeros(size(membership_values,2), 1);
	
	for k = 1: size(aggregatedValues,1)
		aggregatedValue = 1;
		for i = 1: size(membership_values,2)
			aggregatedValue *= membership_values(k,i);
		end;
		
		aggregatedValues(k,1) = aggregatedValue;
	end;
end;


function membership_values = getMembershipValues(input_sample,centroids,sigma)
	membership_values = zeros(size(centroids));
	
	for i = 1 : size(centroids,1)
		membership_values(:,i) = (exp(- (input_sample(:,1) - centroids(:,i) .^ 2)/(2 * sigma ^ 2)  ))';
	end;


end;


function input_sample = getInputSampleWithIndex(sample,index)
	input_sample = zeros(3,1);
	
	test_offset = [6,12,18];
	
	dev_offset = [1,7,14];
	
	input_sample(1) = sample(index - dev_offset(1));
	input_sample(2) = sample(index - dev_offset(2));
	input_sample(3) = sample(index - dev_offset(3));
end;

function X = mackeyGlassOut
	a        = 0.25;     
	b        = 0.1;     
	tau      = 15;		
	x0       = 1.2;		
	deltat   = 0.1;	    
	sample_n = 1200;	
	interval = 1;	  

	time = 0;
	index = 1;
	history_length = floor(tau/deltat);
	x_history = zeros(history_length, 1); 
	x_t = x0;

	X = zeros(sample_n + 1, 1); 
	T = zeros(sample_n + 1, 1); 

	for i = 1:sample_n + 1,
		X(i) = x_t;

		if tau <= 0
			x_t_minus_tau = 0.0;
		else
			x_t_minus_tau = x_history(index);
		end

		x_t_plus_deltat = mackeyglassRungeKutta(x_t, x_t_minus_tau, deltat, a, b);
		if (tau ~= 0),
			x_history(index) = x_t_plus_deltat;
			index = mod(index, history_length) + 1;
		end
		
		time = time + deltat;
		T(i) = time;
		x_t = x_t_plus_deltat;
	end;
end