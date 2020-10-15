% % Author: Fabian Santiago, fsantiago3@ucmerced.edu
% SCRIPT: structured_population_model_log_flux.m
% % This script reproduces the four plots in Figure 6 of the paper.
% % >> The script must be run in its entirety because there are function 
% % definitions at the bottom of the script. 
% %
% % Parameters:
% % alp - cell division rate
% % bet - cell death rate
% % L - constituent replication rate
% % K - constituent carrying capacity
% % 1/gam - division of constituent among resulting daughter cells 
% % 
% % Variables:
% % a(t)    - amount of intracellular constituent at time t
% % da/dt   - intracellular constituent dynamics [ L*a*(1-a/K) ] logistic
% % Yi(t,a) - distribution of intracellular constituents at time t, 
% %           for generation i.   
% % U(a)    - initial distribution of intracellular constituents
% %
% % PDE System:
% % Y0_t+(dy/dt*Y0)_a = -(alp+bet)*Y0                           - 0th gen
% % Yi_t+(da/dt*Yi)_a = -(alp+bet)*Yi+2*alp*gam*Y[i-1](t,gam*a) - ith gen
% % 
% % Recursive Numerical Integration (RNI) Formulation: 
% % Yi(t,y) = 2*gam*alp/Mui(t,S)*int_0^t[Mui(T,S)*Y[i-1](T,gam*y(T,S))]dT
% % 
% % Last Update: 10/14/2020 MATLAB 2020a by Fabian Santiago
clear
close all

% Add the hyperbolic PDE solver to the search space, and use Lax-Wendroff
% for numerical integration
addpath('../solver/')
method = 'LxW';

% Define constant global parameter values
global alp bet mu sig gam lam K
alp = (log(2)/90)*60; bet = 0; % Cellular growth
mu  = 1; sig = 0.1; % Initial Distribution
gam = 2; % Propagons are distributed equally among resulting generation 
M = 7;   % Observe a maximum of 7 generations: N0,N1,...,N7

% Propagon Dynamics Parameter values
lam = 1.5; K = 20;   

% Initial Distribution of Propagons
U  =@(a) normpdf(a,mu,sig);

% Spatial discretization for hyperbolic pde solver
Na1 = 2*10^2; % Dy = 0.1
Na2 = 2*10^3; % Dy = 0.01

% Spatial discretization for hyperbolic pde solver
a_vec1 = linspace(0,K,Na1); % Spatial grid in a with Da = 0.01
a_vec2 = linspace(0,K,Na2); % Spatial grid in a with Da = 0.001

% Pre-allocate space for computation of M generations
Y0_1 = [U(a_vec1);zeros(M,Na1)];
Y0_2 = [U(a_vec2);zeros(M,Na2)];

% Setup hyperbolic pde solver to begin integration at time t = 0
t0 = 0;
solY_Da1 = setup(2,{@FLUX,@SOURCE},t0,a_vec1,Y0_1,method,[],@bcfun);
solY_Da2 = setup(2,{@FLUX,@SOURCE},t0,a_vec2,Y0_2,method,[],@bcfun);

% Guassian Quadrature Points
GQ_pts = [ 3,  3,  3,  3, 4, 3, 4;...
           5,  4,  4,  4, 4, 3, 5;...
          18, 13,  7,  5, 5, 5, 4;...
          54, 33, 22, 13, 7, 7, 5];    

for T = 1:4
    figure
    
    % Solve PDE system using LxW with Da = 0.1 and Da = 0.01
    sol1 = hpde(solY_Da1,T,a_vec1(2)/(lam*(M+1)*K));
    sol2 = hpde(solY_Da2,T,a_vec2(2)/(lam*(M+1)*K));
    
    % Plot both solutions from the pde solver
    subplot(3,2,1:4)
    h1 = plot(sol1.x,sol1.u,'k--','LineWidth',2.5); hold on
    h2 = plot(sol2.x,sol2.u,'k-','LineWidth',2.5);
    
    % Determine the maximum and minimum values observed in numerical
    % solutions for plotting 
    MaxY1 = max([sol1.u(:);sol2.u(:)]); minY1 = min([sol1.u(:);sol2.u(:)]);

    % Load RNI PDE Yi(t,a)
    Y_tmp = Y_lgwt(0,NaN,T,a_vec1);
    Y_RNI = [Y_tmp; zeros(7,numel(a_vec1))];
    plot(a_vec1,Y_tmp,'r-','LineWidth',2.5);
    
    for i = 1:7
       Y_tmp = Y_lgwt(i,GQ_pts(T,i),T,a_vec1);
       h3 = plot(a_vec1,Y_tmp,'r-','LineWidth',2.5);
       Y_RNI(i+1,:) = Y_tmp;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Plotting Options %%%%%%%%%%%%%%%%%%%%%%%%%%%
    xlim([0 K]); ylim([-.5 1.5]);
    set(gca,'Ytick',-.5:.25:1.5)
    set(gca,'Xtick',0:5:K)

    ylabel('Structured Population Density','Interpreter','latex','FontSize',16);
    set(gca,'TickLabelInterpreter','latex','FontSize',16)
    LEG = legend([h1(1),h2(1),h3(1)],{['LxW$_1$: $\Delta ',sprintf(' a=$ %.2f',K/Na1)],...
                ['LxW$_2$: $\Delta ',sprintf(' a=$ %.2f',K/Na2)],'RNI'});
    set(LEG,'Interpreter','latex','FontSize',13,'Location','northeast')
    title(LEG,'$I(a;${\boldmath$\theta$}$)=\lambda a\left(1-\frac{a}{K}\right)$','Interpreter','latex','FontSize',15)
    grid on    
    hold off

    subplot(3,2,5:6)
    Error_LxW1 = (sum(Y_RNI)-sum(sol1.u))/max(Y_RNI(:));
    Error_LxW2 = (sum(Y_RNI)-sum(sol2.u(:,1:10:end)))/max(Y_RNI(:));
    
    minError = min([Error_LxW1 Error_LxW2]);
    maxError = max([Error_LxW1 Error_LxW2]);
    
    plot(a_vec1,Error_LxW1,'k-','LineWidth',2); hold on
    plot(a_vec1,Error_LxW2,'r-','LineWidth',2); hold off
    ylim([-1.5 1.5])
    
    xlabel('Concentration of Intracellular Constituents $(a)$','Interpreter','latex','FontSize',20); 
    ylabel('Error','Interpreter','latex','FontSize',20); 
    
    LEG = legend({'(RNI$-$LxW$_1)/$max(RNI)','(RNI$-$LxW$_2$)$/$max(RNI)'});
    set(LEG,'Interpreter','latex','FontSize',13,'Location','northeast')
    if T>3
        set(LEG,'Interpreter','latex','FontSize',13,'Location','northwest')
    end
    grid on
    axis([0 K -1.3 1.3])
    set(gca,'Xtick',0:5:K)
    set(gca,'Ytick',linspace(-1.3,1.3,5))
    set(gca,'TickLabelInterpreter','latex','FontSize',16)
    set(gcf,'Position', [100, 100, 800, 650])
    shg
end

%%%%%%%%              HYPERBOLIC PDE SOLVER FUNCTIONS              %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Flx = FLUX(t,y,N)
    global L K
    Flx = -L.*y.*(1-y/K).*N;
end

function Src = SOURCE(t,y,N)
    global alp bet gam
    Yi_1 = zeros(size(N));
    Yi_1(2:end,1:floor(end/2)) = N(1:end-1,2:2:end);
    Src = -(alp+bet)*N+2*alp*gam*Yi_1;
end

function [YL,YR] = bcfun(t,NLy,NRy)
    YL = NLy*0; YR = NRy*0;
end

%%%%%%%%             RECURSIVE FUNCTION DEFINITIONS                %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Characteristics
function res = S(t,a)
    global L K
    res =  K.*a./(a-a.*exp(L*t)+K.*exp(L.*t));
end

function res = A(t,s)
    global L K
    res = K.*s.*exp(L*t)./(K-s+s.*exp(L.*t));
end

function res = MuS(t,s)
    global alp bet L K
    res =  (K./(K+s.*(exp(L.*t)-1))).^2.*exp((alp+bet+L).*t);
end

function res = MuA(t,y_vec)
    res = MuS(t,S(t,y_vec));
end

% RNI Solution to DLSPM Model
function Y_out = Y_lgwt(i,P,tf,a_vec)
    global mu alp sig gam
    if i==0
        U  =@(a) normpdf(a,mu,sig);
        Y_out =  U(S(tf,a_vec))./MuA(tf,a_vec);
    elseif i==1
        [T,W] = lgwt(P,0,tf);
        INTGRND = MuS(T,S(tf,a_vec)).*Y_lgwt(i-1,P,T,gam*A(T,S(tf,a_vec)));
        Y_out = 2*gam*alp./MuS(tf,S(tf,a_vec)).*sum(W.*INTGRND);
    else
        [T,W] = lgwt(P,0,tf);
        INTGRND = zeros(P,numel(a_vec));
        for i = 1:P
            INTGRND(i,:) = MuS(T(i),S(tf,a_vec)).*Y_lgwt(i-1,P,T(i),gam*A(T(i),S(tf,a_vec)));
        end
        Y_out = 2*gam*alp./MuS(tf,S(tf,a_vec)).*sum(W.*INTGRND);
    end
    
end
