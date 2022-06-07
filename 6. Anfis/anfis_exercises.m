%% Sistemas Nebulosos
% Vítor Gabriel Reis Caitité - 2021712430

%% QUESTÃO 1: Modelagem de sistema estático monovariável
%
% Aproximar a função y=x^2
%_____________________________________________________________________

close all; clear; clc;
warning('off','all');

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
fig_number = 1;
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'GridPartition',2);
figure(fig_number)
plot(X_test, y_test, X_test, ys)
legend('Test Data','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using Subtractive Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'SubtractiveClustering');
figure(fig_number)
plot(X_test, y_test, X_test, ys)
legend('Test Data','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using FCM Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'FCMClustering');
figure(fig_number)
plot(X_test, y_test, X_test, ys)
legend('Test Data','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;



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
figure(fig_number)
plot(y_train);
title("y_{train}");
drawnow();
fig_number = fig_number + 1;
figure(fig_number)
plot(y_test);
title("y_{test}");
drawnow();
fig_number = fig_number + 1;

%% Generate FIS Using Grid Partitioning
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'GridPartition',2);
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using Subtractive Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'SubtractiveClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using FCM Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'FCMClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;



%% QUESTÃO 3: Modelo de sistema dinâmico
%_____________________________________________________________________
X_train = table2array(readtable('ex3_X_train.csv'));
y_train = table2array(readtable('ex3_y_train.csv'));
X_test = table2array(readtable('ex3_X_test.csv'));
y_test = table2array(readtable('ex3_y_test.csv'));
figure(fig_number)
plot(y_train);
title("y_{train}");
drawnow();
fig_number = fig_number + 1;
figure(fig_number)
plot(y_test);
title("y_{test}");
drawnow();
fig_number = fig_number + 1;

%% Generate FIS Using Grid Partitioning
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'GridPartition',  2);
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using Subtractive Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'SubtractiveClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using FCM Clustering
[ys, ERROR] =run_anfis(X_train, y_train, X_test, 'FCMClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;



%% QUESTÃO 4: Previsão de uma série temporal caótica
%_____________________________________________________________________
X_train = table2array(readtable('ex4_X_train.csv'));
y_train = table2array(readtable('ex4_y_train.csv'));
X_test = table2array(readtable('ex4_X_test.csv'));
y_test = table2array(readtable('ex4_y_test.csv'));
figure(fig_number)
plot(y_train);
title("y_{train}");
drawnow();
fig_number = fig_number + 1;
figure(fig_number)
plot(y_test);
title("y_{test}");
drawnow();
fig_number = fig_number + 1;

%% Generate FIS Using Grid Partitioning
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'GridPartition',  2);
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using Subtractive Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'SubtractiveClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using FCM Clustering
[ys, ERROR] =run_anfis(X_train, y_train, X_test, 'FCMClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;


%% QUESTÃO 5: Data set UCI
%_____________________________________________________________________
X_train = table2array(readtable('ex5_X_train.csv'));
y_train = table2array(readtable('ex5_y_train.csv'));
X_test = table2array(readtable('ex5_X_test.csv'));
y_test = table2array(readtable('ex5_y_test.csv'));
figure(fig_number)
plot(y_train);
title("y_{train}");
drawnow();
fig_number = fig_number + 1;
figure(fig_number)
plot(y_test);
title("y_{test}");
drawnow();
fig_number = fig_number + 1;

%% Generate FIS Using Grid Partitioning
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'GridPartition',  2);
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using Subtractive Clustering
[ys, ERROR] = run_anfis(X_train, y_train, X_test, 'SubtractiveClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;

%% Generate FIS Using FCM Clustering
[ys, ERROR] =run_anfis(X_train, y_train, X_test, 'FCMClustering');
figure(fig_number)
plot(y_test)
hold on
plot(ys)
legend('y_{test}','Anfis Output');
drawnow();
figure(fig_number+1)
plot(ERROR.^2)
drawnow();
fprintf('MSE: %.2E', immse(ys,y_test));
fig_number = fig_number + 2;