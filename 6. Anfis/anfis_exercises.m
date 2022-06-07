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
drawnow();
figure(2)
plot(ERROR.^2)
fprintf('MSE: %.2E', immse(ys,y_test));
drawnow();


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
plot(X_train, y_train, X_test, ys);
legend('Training Data','Anfis Output');
drawnow();
figure(4)
plot(ERROR.^2);
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));

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
plot(X_train, y_train, X_test, ys);
legend('Training Data','Anfis Output');
drawnow();
figure(6)
plot(ERROR.^2);
fprintf('MSE: %.2E', immse(ys,y_test));
drawnow();

%% QUESTÃO 2: Modelagem de sistema estático multivariável
%
% Modelar uma função não linear de 3 entradas:
%
% output = (1 + x^0.5 + y^-1 + z^-1.5)^2
%_____________________________________________________________________
X_train = table2array(readtable('ex2_X_train.csv'));
y_train = table2array(readtable('ex2_y_train.csv'));
X_test = table2array(readtable('ex2_X_test.csv'));
y_test = table2array(readtable('ex2_y_test.csv'));
figure(7)
plot(y_train);
title("y_{train}");
drawnow();
figure(8)
plot(y_test);
title("y_{test}");
drawnow();

%% Generate FIS Using Grid Partitioning
options = genfisOptions('GridPartition');
options.NumMembershipFunctions = 3;
in_fis = genfis(X_train,y_train,options);

options = anfisOptions;
options.InitialFIS = in_fis;
options.EpochNumber = 100;
options.DisplayStepSize = 0;
options.DisplayErrorValues = 0;
[out_fis,ERROR] = anfis([X_train y_train],options);
ys=evalfis(out_fis, X_test);
figure(9)
plot(y_test);
hold on
plot(ys);
legend('Test Data','Anfis Output');
drawnow();
figure(10)
plot(ERROR.^2);
drawnow();
fprintf('MSE: %.2E',  immse(ys,y_test));



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
figure(11)
plot(y_test);
hold on
plot(ys);
drawnow();
legend('Test Data','Anfis Output');
figure(12)
plot(ERROR.^2);
drawnow();
fprintf('MSE: %.2E',  immse(ys,y_test));

%% Generate FIS Using FCM Clustering

options = genfisOptions('FCMClustering');
options.Verbose = false;
in_fis = genfis(X_train,y_train, options);

options = anfisOptions;
options.InitialFIS = in_fis;
options.EpochNumber = 100;
options.DisplayStepSize = 0;
options.DisplayErrorValues = 0;
[out_fis,ERROR] = anfis([X_train y_train], options);
ys=evalfis(out_fis, X_test);
figure(13)
plot(y_test);
hold on
plot(ys);
legend('Test Data','Anfis Output');
drawnow();
figure(14)
plot(ERROR.^2);
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));


