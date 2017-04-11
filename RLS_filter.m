clear all
close all
warning off

fs=44100;
d = audioread('rolling_noise.wav');
x = audioread('Heli_noise.wav');
d = d(1:2*fs,:);
x = x(1:2*fs,:);
filterOrder = 6;
forgettingFactor = 1;
delta = 1/664;
[e_RLS(:,1), h_RLS(:,:,1)] = wienerFilterRLS(x(:,1), d(:,1), filterOrder, forgettingFactor, delta);
[e2_RLS, h2_RLS] = wienerFilterRLS  (x(:,2), d(:,2), filterOrder, forgettingFactor, delta);
soundsc(e_RLS, fs);

% Desired Signal Plot
%     dt = 100/fs;
%     t = 0:dt:(length(e_RLS)*dt)-dt;
%     plot(t,e_RLS); xlabel('Seconds'); ylabel('Amplitude')
%     axis([0 inf -2 2]);


function [e, h] = wienerFilterRLS(x, d, filterOrder, lambda, delta)
%WIENERFILTERRLS Process a signal using the Wiener filter using the
%Recursive Least Squares (RLS) method
%   INPUTS:
%    - x: noise
%    - d: desired signal with noise
%    - filterOrder: amount of past signals used (one less than the total
%    number of coefficients)
%    - lambda: forgetting factor of the RLS algorithm
%    - delta: value to initialize the P matrix
%   OUTPUTS:
%    - e: desired signal without the noise
%    - h: coefficients of the Wiener filter at each step

% Initialization
nCoefficients = filterOrder + 1;
X = tril(toeplitz(x, zeros(1, nCoefficients)));
Pbefore = (1/delta)*eye(nCoefficients);

% Memory allocation
hbefore = zeros(nCoefficients, 1);
h = zeros(nCoefficients, length(x));
yEstimated = zeros(length(x),1);
e = zeros(length(x),1);

% RLS algorithm and Wiener Filter
for i = 1:length(x)
    % RLS algorithm
    Xi = X(i,:)';
    a = d(i) - Xi'*hbefore;
    g = (Pbefore*Xi)/(lambda+Xi'*Pbefore*Xi);
    P = (1/lambda)*(Pbefore-g*Xi'*Pbefore);
    h(:,i) = hbefore + a*g;
    
    % P and h shifting for next cycle
    Pbefore = P;
    hbefore = h(:,i);
    
    % Wiener Filter
    yEstimated(i) = Xi'*h(:,i);
    
    % Desired signal computation
    e(i) = d(i) - yEstimated(i);
end

end
