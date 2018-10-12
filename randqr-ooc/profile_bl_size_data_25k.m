% n = 25000
b = [100 128 256 400 512 600];

perc_comm = [49.12 43.65 25.52 15.57 11.15 9.20];

perc_flop = [50.88 56.35 74.48 84.43 88.85 90.80];

t_tot = [3.321628e+02 2.956317e+02 2.590132e+02 2.802799e+02 2.964598e+02 3.226719e+02];

t_comm = [1.631727e+02 1.290446e+02 6.609833e+01 4.364273e+01 3.306157e+01 2.967446e+01];

t_flop = [1.689901e+02 1.665870e+02 1.929149e+02 2.366371e+02 2.633982e+02 2.929974e+02];

% Working on b = 100; 1/6 
% t_comm:          1.631727e+02    49.12%
%   t_read:          6.481776e+01    19.51%
%   t_write:         9.835496e+01    29.61%
% t_flop:          1.689901e+02    50.88%
%   t_init_Y:        1.552015e+00    0.47%
%   t_qr_Y:          1.563830e+01    4.71%
%   t_qr_A:          9.582722e+00    2.88%
%   t_update_A:      1.404900e+02    42.30%
%   t_downdate_Y:    1.727015e+00    0.52%
% total_time:          3.321628e+02
% Working on b = 128; 2/6 
% t_comm:          1.290446e+02    43.65%
%   t_read:          5.107401e+01    17.28%
%   t_write:         7.797063e+01    26.37%
% t_flop:          1.665870e+02    56.35%
%   t_init_Y:        1.664697e+00    0.56%
%   t_qr_Y:          2.389525e+01    8.08%
%   t_qr_A:          1.305724e+01    4.42%
%   t_update_A:      1.260329e+02    42.63%
%   t_downdate_Y:    1.936945e+00    0.66%
% total_time:          2.956317e+02
% Working on b = 256; 3/6 
% t_comm:          6.609833e+01    25.52%
%   t_read:          2.674326e+01    10.33%
%   t_write:         3.935507e+01    15.19%
% t_flop:          1.929149e+02    74.48%
%   t_init_Y:        2.360334e+00    0.91%
%   t_qr_Y:          4.667773e+01    18.02%
%   t_qr_A:          2.968943e+01    11.46%
%   t_update_A:      1.107970e+02    42.78%
%   t_downdate_Y:    3.390412e+00    1.31%
% total_time:          2.590132e+02
% Working on b = 400; 4/6 
% t_comm:          4.364273e+01    15.57%
%   t_read:          1.759210e+01    6.28%
%   t_write:         2.605062e+01    9.29%
% t_flop:          2.366371e+02    84.43%
%   t_init_Y:        3.273887e+00    1.17%
%   t_qr_Y:          6.475374e+01    23.10%
%   t_qr_A:          5.071534e+01    18.09%
%   t_update_A:      1.125163e+02    40.14%
%   t_downdate_Y:    5.377847e+00    1.92%
% total_time:          2.802799e+02
% Working on b = 512; 5/6 
% t_comm:          3.306157e+01    11.15%
%   t_read:          1.408901e+01    4.75%
%   t_write:         1.897256e+01    6.40%
% t_flop:          2.633982e+02    88.85%
%   t_init_Y:        3.908821e+00    1.32%
%   t_qr_Y:          8.455165e+01    28.52%
%   t_qr_A:          6.718369e+01    22.66%
%   t_update_A:      1.013115e+02    34.17%
%   t_downdate_Y:    6.442532e+00    2.17%
% total_time:          2.964598e+02
% Working on b = 600; 6/6 
% t_comm:          2.967446e+01    9.20%
%   t_read:          1.227543e+01    3.80%
%   t_write:         1.739903e+01    5.39%
% t_flop:          2.929974e+02    90.80%
%   t_init_Y:        4.391473e+00    1.36%
%   t_qr_Y:          9.655479e+01    29.92%
%   t_qr_A:          8.026849e+01    24.88%
%   t_update_A:      1.041767e+02    32.29%
%   t_downdate_Y:    7.605964e+00    2.36%
% total_time:          3.226719e+02
% End of Program
