% Lire le fichier en conservant les noms originaux
data = readmatrix(['data.ods']);
Te = data(1,1);
Tf = max(data(:,1))

% Extraire les colonnes
time = data(:,1) -data(1,1) ;  % colonne temps
C_ref = data(:,2);  % colonne setpoint
C_mes = data(:,3);  % colonne feedback

% Vérifie qu'il n'y a pas de NaN ou Inf
% C_ref(isnan(C_ref)) = 0;  
% C_ref(isinf(C_ref)) = 0;
% 
% C_mes(isnan(C_mes)) = 0;
% C_mes(isinf(C_mes)) = 0;

% Créer des timeseries pour Simulink
C_ref_d = [time C_ref];
C_mes_d = [time C_mes];


figure(1)
plot(time, C_mes, 'b');
xlabel('Time (s)');
ylabel('Values');
legend('Setpoint');
title('Setpoint vs Feedback');
grid on
hold on
plot(time,ye_mcr, 'r')