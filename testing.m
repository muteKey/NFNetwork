
function testing()
a        = 0.25;     
b        = 0.1;     
tau      = 15;		
x0       = 1.2;		
deltat   = 0.1;	    
sample_n = 12000;	
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
    if (mod(i - 1, interval) == 0),
        disp(sprintf('%4d %f', (i - 1)/interval, x_t));
    end
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

figure
plot(T, X);
set(gca,'xlim',[0, T(end)]);
xlabel('t');
ylabel('x(t)');
title(sprintf('A Mackey-Glass time series (tau=%d)', tau));

end;  
