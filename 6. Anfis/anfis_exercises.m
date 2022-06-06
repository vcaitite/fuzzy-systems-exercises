%% TRABALHO PRÁTICO - Sistemas Nebulosos
% Vítor Gabriel Reis Caitité - 2016111849 



%% QUESTÃO 1: Modelagem de sistema estático monovariável
%
% Aproximar a função y=x^2
%_____________________________________________________________________

close all; clear; clc;

% Geração de dados
N = 1000;
X = (linspace(-2, 2, N)).';
y = (X.^2);

idx = randperm(length(y));
X_train = X(sort(idx(1:900)));
y_train = y(sort(idx(1:900)));
X_test = X(sort(idx(901:1000)));
y_test = y(sort(idx(901:1000)));

%% Generate FIS Using Grid Partitioning
options = genfisOptions('GridPartition');
options.NumMembershipFunctions = 2;
in_fis = genfis(X_train,y_train,options);

options = anfisOptions;
options.InitialFIS = in_fis;
options.EpochNumber = 100;
options.DisplayStepSize = 0;
options.DisplayErrorValues = 0;
[out_fis,ERROR] = anfis([X_train y_train],options);
ys=evalfis(out_fis, X_test);
figure(1)
plot(X_train, y_train, X_test, ys)
legend('Training Data','Anfis Output');
figure(2)
plot(ERROR.^2)
fprintf('MSE: %.2E', ERROR(20)^2);


%% Generate FIS Using Subtractive Clustering

options = genfisOptions('SubtractiveClustering');
in_fis = genfis(X_train,y_train,options);

options = anfisOptions;
options.InitialFIS = in_fis;
options.EpochNumber = 100;
options.DisplayStepSize = 0;
options.DisplayErrorValues = 0;
[out_fis,ERROR] = anfis([X_train y_train],options);
ys=evalfis(out_fis, X_test);
figure(3)
plot(X_train, y_train, X_test, ys)
legend('Training Data','Anfis Output');
figure(4)
plot(ERROR.^2)
fprintf('MSE: %.2E', ERROR(20)^2);

%% Generate FIS Using FCM Clustering

options = genfisOptions('FCMClustering');
options.Verbose = false;
in_fis = genfis(X_train,y_train,options);

options = anfisOptions;
options.InitialFIS = in_fis;
options.EpochNumber = 100;
options.DisplayStepSize = 0;
options.DisplayErrorValues = 0;
[out_fis,ERROR] = anfis([X_train y_train],options);
ys=evalfis(out_fis, X_test);
figure(5)
plot(X_train, y_train, X_test, ys)
legend('Training Data','Anfis Output');
figure(6)
plot(ERROR.^2)
fprintf('MSE: %.2E', ERROR(20)^2);








