function x_dot = mackeyglass(x_t, x_t_minus_tau, par_a, par_b)
    x_dot = -par_b * x_t + par_a * x_t_minus_tau / (1 + x_t_minus_tau ^ 10.0);
end
