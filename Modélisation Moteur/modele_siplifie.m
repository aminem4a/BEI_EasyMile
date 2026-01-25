%% --- Lecture des nouvelles données simplifiées ---
data_ref_simpli  = readmatrix('modeli_simplifie.xlsx', 'Sheet', 'reff');
data_feed_simpli = readmatrix('modeli_simplifie.xlsx', 'Sheet', 'feedB');
data_ref = readmatrix('Motor_StepResponse-1.xlsx', 'Sheet', 'Axle_torque_setpoint');
data_feed = readmatrix('Motor_StepResponse-1.xlsx', 'Sheet', 'Axle_torque_feedback');
%% --- Extraction des signaux ---
time_ref  = data_ref_simpli(:,7)  - data_ref_simpli(1,7); % temps pour setpoint
C_ref     = data_ref_simpli(:,8);                        % setpoint

time_feed = data_feed_simpli(:,2) - data_feed_simpli(1,2); % temps pour feedback
C_feed    = data_feed_simpli(:,5);                          % feedback
C_feed_d  = [time_feed C_feed];
C_ref_d = [time_ref  C_ref];

%% signaux complets
time_ref_full  = (data_ref(:,7)  - data_ref_simpli(1,7)); % temps pour setpoint
C_ref_full     = data_ref(:,8);                        % setpoint

time_feed_full = (data_feed(:,2) - data_feed_simpli(1,2)); % temps pour feedback
C_feed_full    = data_feed(:,5);                          % feedback
C_feed_full_d  = [time_feed_full C_feed_full];
C_ref_full_d = [time_ref_full  C_ref_full];
%% --- Tracé des signaux bruts ---
figure;
plot(time_ref, C_ref, 'b', 'LineWidth',1.5); hold on;
plot(time_feed, C_feed, 'r', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Amplitude');
title('Setpoint et Feedback');
legend('C_{ref}','C_{feed}');
grid on;

%% --- Détection du front de step ---
dy = diff(C_ref);
[~, idx_step] = max(abs(dy));

if idx_step + 1 <= length(time_ref)
    t_step = time_ref(idx_step + 1);
else
    t_step = time_ref(end);
end

%% --- Définition de la fenêtre autour du step ---
t_before = 0.2;  % avant le step
t_after  = 2.0;  % après le step
mask = (time_feed >= t_step - t_before) & (time_feed <= t_step + t_after);

if sum(mask) < 2
    warning('Fenêtre trop courte, on prend tous les points après le step.');
    mask = (time_feed >= t_step);
end

t  = time_feed(mask) - time_feed(find(mask,1));
yf = C_feed(mask);
yr = C_ref(mask);


%% --- Méthode de Strejc ---
k = yf(end)/yr(end);          % gain statique
dt = t(2) - t(1);     % pas de temps

% dérivée numérique
dyf = diff(yf)/dt;
dyf = [dyf;0];        % garder la même longueur

[dyfmax, imax] = max(dyf); % point d'inflexion
tmax = t(imax);
ypi = yf(imax);

% tangente
a = dyfmax;
b = ypi - a*tmax;
tan_line = a*t + b;

% intersections tangente
t1 = -b/a;
t2 = (k - b)/a;
Td = t1;
Ta = t2 - t1;

i = find(yf~=0,1);
Tr = t(i);

Tu = Td - Tr;
rapport = Tu / Ta;
n = 2;  % valeur typique

tau = Ta / exp(1);
Tutab = Ta * 0.104;
TR = Tu - Tutab;
T = Tr + TR;
if T < 0, T = 0; end

%% --- Affichage résultats ---
fprintf("Résultats du modèle Strejc pour les données simplifiées :\n");
fprintf("k       = %.4f\n", k);
fprintf("Td      = %.4f\n", Td);
fprintf("Ta      = %.4f\n", Ta);
fprintf("n       = %d\n", n);
fprintf("tau     = %.4f\n", tau);
fprintf("T       = %.4f\n", T);
fprintf("rapport = %.4f\n", rapport);

%% --- Création des timeseries pour Simulink ---
C_ref_ts  = timeseries(yr, t);
C_feed_ts = timeseries(yf, t);


%% --- Simulation de la réponse du modèle avec retard ---
s = tf('s');                        % variable de Laplace

F = k * exp(-T*s) / ( (tau*s + 1)^n );  % fonction de transfert Strejc

% Vecteur temps positif et suffisant pour observer la réponse
t_sim = 0:dt:5*(tau + T);  % 5*T typique pour que le step atteigne l'état stable



% Facteurs pour la fonction de transfert de simulink
tau1 = 30;
r = 50.65;



% step accepte les systèmes avec retard
[y_sim, t_sim] = step(F, t_sim);

