% SCRIPT: structured_population_model_log_flux.m
% % This script compares Lax-Wendroff (LxW) to Recursive Numerical 
% % Integration (RNI) for solving a structured population model system of 
% % PDEs. RNI depends on a recursive integral solution of the system of
% % PDES while Lax-Wendroff solves the system of PDEs on a prescribed
% % spatial grid. The final output is the figure included in the writeup.
% % On a 3.40 GHz quad-core, 64 GB Ram system the script takes about 4 hours
% % to run.
% % >> The script must be run in its entirety because there are function 
% % definitions at the bottom of the script. 
% %
% % Parameters:
% % L - propagon replication rate
% % K - propagon carrying capacity
% % gam - 1/propagon division among resulting daughter cells 
% % 
% % Variables:
% % y(t)    - propagons at time t
% % dy/dt   - propagon dynamics [ L*a*(1-a/K) ] logistic
% % Ni(t,y) - propagon distribution at time t, for generation i   
% % U(y)    - initial distribution of propagons
% %
% % PDE System:
% % N0_t+(dy/dt*N0)_a = -(alp+bet)*N0                           - 0th gen
% % Ni_t+(da/dt*Ni)_a = -(alp+bet)*Ni+2*alp*gam*P[i-1](t,gam*y) - ith gen
% % 
% % Recursive Numerical Integration (RNI) Formulation: 
% % Ni(t,y) = 2*gam*alp/Mui(t,S)*int_0^t[Mui(T,S)*P[i-1](T,gam*y(T,S))]dT
% % 
% % Author: Fabian Santiago
% % Last Update: 1/29/19 MATLAB 2019a

% close all figures and clear all variable variables in the Workspace 
close all
clear

% Add the hyperbolic PDE solver to the search space, and use Lax-Wendroff
% for numerical integration
addpath('solver/')
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
Na1 = 2*10^3; % Dy = 0.01
Na2 = 2*10^4; % Dy = 0.001
y_vec1  = linspace(0,K,Na1); % Spatial grid in y with Dy = 0.01
y_vec2  = linspace(0,K,Na2); % Spatial grid in y with Dy = 0.001

% Pre-allocate space for computation of M generations
N0_1 = [U(y_vec1);zeros(M,Na1)];
N0_2 = [U(y_vec2);zeros(M,Na2)];

% Setup hyperbolic pde solver to begin integration at time t = 0
t0 = 0;
solY_Da1 = setup(2,{@FLUX,@SOURCE},t0,y_vec1,N0_1,method,[],@bcfun);
solY_Da2 = setup(2,{@FLUX,@SOURCE},t0,y_vec2,N0_2,method,[],@bcfun);

% Solve PDE System Using Lax-Wendroff with Two Refinements and Using
% Recursive Numerical Integration for Different Time Values and Plot the
% Solution for each time value.
for T = 2
    figure

    % Solve PDE system using LxW with Da = 0.01 and Da = 0.001
    sol1 = hpde(solY_Da1,T,y_vec1(2)/(lam*(M+1)*K));
    sol2 = hpde(solY_Da2,T,y_vec2(2)/(lam*(M+1)*K));
    
    % Plot both solutions from the pde solver
    subplot(3,2,1:4)
    for g = 1:(M+1)
        h1 = plot(sol1.x,sol1.u(g,:),'k--','LineWidth',2.5); hold on
    end   
    for g = 1:(M+1)
        h2 = plot(sol2.x,sol2.u(g,:),'k-','LineWidth',2.5);
    end

    % Determine the maximum and minimum values observed in numerical
    % solutions for plotting 
    Maxy1 = max(sol1.u(:));    miny1 = min(sol1.u(:));
    Maxy2 = max(sol2.u(:));    miny2 = min(sol2.u(:));    

    % Solve for each generation Ni(t,y) using RNI
    N1tmp = N1(T,y_vec1,200);
    N2tmp = N2(T,y_vec1,100);
    N3tmp = N3(T,y_vec1,50);
    N4tmp = N4(T,y_vec1,25);
    N5tmp = N5(T,y_vec1,14);
    N6tmp = N6(T,y_vec1,8);
    N7tmp = N7(T,y_vec1,5);
    
    % Place all generations in one matrix
    Y_RNI = [N0(T,y_vec1);N1tmp;N2tmp;N3tmp;N4tmp;N5tmp;N6tmp;N7tmp];
    
    % plot RNI solutions to PDE
    plot(y_vec1,N0(T,y_vec1),'r-','LineWidth',2.5);
    plot(y_vec1,N1tmp,'r-','LineWidth',2.5);
    plot(y_vec1,N2tmp,'r-','LineWidth',2.5);
    plot(y_vec1,N3tmp,'r-','LineWidth',2.5);    
    plot(y_vec1,N4tmp,'r-','LineWidth',2.5);
    plot(y_vec1,N5tmp,'r-','LineWidth',2.5);
    plot(y_vec1,N6tmp,'r-','LineWidth',2.5);
    h3 = plot(y_vec1,N7tmp,'r-','LineWidth',2.5); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Plotting Options %%%%%%%%%%%%%%%%%%%%%%%%%%%
    set(gca,'Xtick',0:5:K)
    max_Y = max([Maxy1 Maxy2]);
    min_Y = min([miny2 miny1]);
    xlim([0 K]); ylim([min_Y max_Y+max_Y/50]);
    grid on
    title([sprintf('$T=$ %.2f',T),' hrs'],'Interpreter','latex','FontSize',20)

    ylabel('Structured Population Density','Interpreter','latex','FontSize',20);
    set(gca,'TickLabelInterpreter','latex','FontSize',20)
    LEG = legend([h1,h2,h3],{'LxW$_1$: $\Delta y= 0.01$','LxW$_2$: $\Delta y= 0.001$','RNI'});
    set(LEG,'Interpreter','latex','FontSize',20,'Location','northeast')
    hold off
    subplot(3,2,5:6)
    
    plot(y_vec1,sum(Y_RNI)-sum(sol1.u),'k-','LineWidth',2);hold on
    plot(y_vec1,sum(Y_RNI)-sum(sol2.u(:,1:10:end)),'r-','LineWidth',2); hold off
    xlabel('Number of Propagons Per Cell','Interpreter','latex','FontSize',20); 
    ylabel('Error','Interpreter','latex','FontSize',20); 
    LEG = legend({'RNI$-$LxW$_1$','RNI$-$LxW$_2$'});
    set(LEG,'Interpreter','latex','FontSize',20,'Location','southeast')
    grid on
    set(gca,'Xtick',0:5:K)   
    set(gca,'TickLabelInterpreter','latex','FontSize',20)
    shg
end
% END SCRIPT

%%%%%%%%              HYPERBOLIC PDE SOLVER FUNCTIONS              %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Flx = FLUX(t,y,N)
    global lam K
    Flx = -lam.*y.*(1-y/K).*N;
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
    global lam K
    res =  K.*a./(a-a.*exp(lam*t)+K.*exp(lam.*t));
end

function res = A(t,s)
    global lam K
    res = K.*s.*exp(lam*t)./(K-s+s.*exp(lam.*t));
end

function res = MuS(t,s)
    global alp bet lam K
    res =  (K./(K+s.*(exp(lam.*t)-1))).^2.*exp((alp+bet+lam).*t);
end

function res = MuA(t,y_vec)
    res = MuS(t,S(t,y_vec));
end

% Recursive Generation Definitions
function N1_out = N0(tf,y_vec)
    global mu sig
    U  =@(y) normpdf(y,mu,sig);
	N1_out =  U(S(tf,y_vec))./MuA(tf,y_vec);
end

function N1_out = N1(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = MuS(T,S(tf,y_vec)).*N0(T,gam*A(T,S(tf,y_vec)));
    N1_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N2_out = N2(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N1(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N2_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N3_out = N3(tf,y_vec,P)
global alp gam 
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N2(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N3_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N4_out = N4(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N3(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N4_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N5_out = N5(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N4(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N5_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N6_out = N6(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N5(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N6_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end

function N7_out = N7(tf,y_vec,P)
global alp gam
    [T,W] = lgwt(P,0,tf);
    INTGRND = zeros(P,numel(y_vec));
    for i = 1:P
        INTGRND(i,:) = MuS(T(i),S(tf,y_vec)).*N6(T(i),gam*A(T(i),S(tf,y_vec)),P);
    end
    N7_out = 2*gam*alp./MuS(tf,S(tf,y_vec)).*sum(W.*INTGRND);
end