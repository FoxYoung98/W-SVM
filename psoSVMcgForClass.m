function [bestCVaccuarcy,bestc,bestg,pso_option] = psoSVMcgForClass(train_label,train,pso_option)
% psoSVMcgForClass

%% Parameter initialization
if nargin == 2
    pso_option = struct('c1',1.5,'c2',1.7,'maxgen',10,'sizepop',20, ...
        'k',0.6,'wV',1,'wP',1,'v',3, ...
        'popcmax',10^2,'popcmin',10^(-1),'popgmax',10^3,'popgmin',10^(-2));
end

% c1: Initially 1.5, pso parameter local search capability
% c2: Initially 1.7, global search capability for pso parameters
% maxgen: starts at 200, maximum number of evolutions
% sizepop: Starts at 20, maximum population size
% k: starts with 0.6 (k belongs to [0.1,1.0]), relationship between rate and x (V = kX)
% wV: Initially 1(wV best belongs to [0.8,1.2]), rate updates the elastic coefficient before velocity in the formula
% wP: Starting with 1, the elastic coefficient before velocity in the population update formula
% v: Initially 3,SVM Cross Validation parameter
% popcmax: Initially 100, the maximum variation of the SVM parameter c.
% popcmin: Initially 0.1, the minimum change in the SVM parameter c.
% popgmax: Initially 1000, the maximum variation of the SVM parameter g.
% popgmin: Initially 0.01, the minimum change in the SVM parameter g.

Vcmax = pso_option.k*pso_option.popcmax;
Vcmin = -Vcmax ;
Vgmax = pso_option.k*pso_option.popgmax;
Vgmin = -Vgmax ;

eps = 10^(-3);

%% Generate initial particles and velocities
for i=1:pso_option.sizepop
    
    % Random generation of population and speed
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
    pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
    V(i,1)=Vcmax*rands(1,1);
    V(i,2)=Vgmax*rands(1,1);
    
    % Calculate initial fitness
    cmd = ['-v ',num2str(pso_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
    fitness(i) = svmtrain(train_label, train, cmd);
    fitness(i) = -fitness(i);
end

% Find extreme values and extreme points
[global_fitness bestindex]=min(fitness); % Global extremum
local_fitness=fitness;   % Individual extremum initialization

global_x=pop(bestindex,:);  % Global extreme point
local_x=pop;    % Individual extreme point initialization

% The average fitness of each generation
avgfitness_gen = zeros(1,pso_option.maxgen);

%% Iteratively search for the best parameters
for i=1:pso_option.maxgen
    
    for j=1:pso_option.sizepop
        
        % Speed update
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        if V(j,2) > Vgmax
            V(j,2) = Vgmax;
        end
        if V(j,2) < Vgmin
            V(j,2) = Vgmin;
        end
        
        % Population regeneration
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        if pop(j,1) > pso_option.popcmax
            pop(j,1) = pso_option.popcmax;
        end
        if pop(j,1) < pso_option.popcmin
            pop(j,1) = pso_option.popcmin;
        end
        if pop(j,2) > pso_option.popgmax
            pop(j,2) = pso_option.popgmax;
        end
        if pop(j,2) < pso_option.popgmin
            pop(j,2) = pso_option.popgmin;
        end
        
        % Adaptive particle variation
        if rand>0.5
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (20-1)*rand+1;
            end
            if k == 2
                pop(j,k) = (pso_option.popgmax-pso_option.popgmin)*rand + pso_option.popgmin;
            end
        end
        
        % Fitness value
        cmd = ['-v ',num2str(pso_option.v),' -c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) )];
        fitness(j) = svmtrain(train_label, train, cmd);
        
        fitness(j) = -fitness(j);
        
        cmd_temp = ['-c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) )];
        model = svmtrain(train_label, train, cmd_temp);
        
        if fitness(j) >= -65
            continue;
        end
        
        % Individual optimal renewal
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if abs( fitness(j)-local_fitness(j) )<=eps && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        % Group optimal renewal
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
        if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i) = global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% Result analysis
figure;
hold on;
plot(-fit_gen,'r*-','LineWidth',1.5);
plot(-avgfitness_gen,'bo-','LineWidth',1.5);
legend('Best fitness','Average fitness');
xlabel('Evolution algebra','FontSize',12);
ylabel('Fitness','FontSize',12);
grid on;

% print -dtiff -r600 pso

bestc = global_x(1);
bestg = global_x(2);
bestCVaccuarcy = -fit_gen(pso_option.maxgen);

line3 = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestCVaccuarcy),'%'];

title({line3},'FontSize',12);

% Print selection
disp('Print selection');
str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestCVaccuarcy,bestc,bestg);
disp(str);
