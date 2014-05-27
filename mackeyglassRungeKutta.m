function x_t_plus_deltat = mackeyglassRungeKutta(x_t, x_t_minus_tau, deltat, a, b)
    k1 = deltat*mackeyglass(x_t,          x_t_minus_tau, a, b);
    k2 = deltat*mackeyglass(x_t+0.5*k1,   x_t_minus_tau, a, b);
    k3 = deltat*mackeyglass(x_t+0.5*k2,   x_t_minus_tau, a, b);
    k4 = deltat*mackeyglass(x_t+k3,       x_t_minus_tau, a, b);
    x_t_plus_deltat = (x_t + k1/6 + k2/3 + k3/3 + k4/6);
end