
%% ============================================================
%  1D fully-implicit frozen-coefficient scheme (matrix left-division)
%  Add: relative error curves of mass and energy
%  Add (NEW): distance between x-location of max(U) and min(U) vs time -> figure(2)
%     d(t) = | x_max(t) - x_min(t) |
% ============================================================

clear; clc; close all;

%% ------------------ Parameters ------------------
k1    = -1.04;
k2    = 0.96;

 % zeta1 = 2.5;%多个
 % zeta2 = 2.5;
 % zeta1 = 3.0;%多个振动,更少了一些
 % zeta2 = 3.0;
  zeta1 = 4.0;%一个振动
  zeta2 = -4.0;

xL = -15;   xR = 35;
dx = 0.02;

%T  = 10.8925*2;

t0 = 6.75; %Collision time
T  = 20;   % final time
dt = 0.005;

% grid
x  = (xL:dx:xR).';         % column, length L
L  = numel(x);
Nt = floor(T/dt) + 1;
t  = (0:dt:(Nt-1)*dt);     % row, length Nt

fprintf('Grid: L=%d, Nt=%d, dx=%.3g, dt=%.3g\n', L, Nt, dx, dt);

% safety checks
if L < 7
    error('Need L >= 7 so that l-3..l+3 stencils exist.');
end
if (L-3) < 4
    error('No interior points: need L large enough so 4..L-3 is non-empty.');
end

%% ------------------ Initial condition (n=1 => t=0) ------------------
U = k1 * sech(k1 *( x + 1*zeta1)) + k2 * sech(k2 *( x + 1*zeta2));

% enforce boundary l=1,2,3 and L-2,L-1,L
U(1:3)   = 0;
U(L-2:L) = 0;

% store history
Uhist = zeros(L, Nt);
Uhist(:,1) = U;

% conservation diagnostics
mass   = zeros(Nt,1);   % \int u dx
energy = zeros(Nt,1);   % \int u^2 dx
mass(1)   = simpson1d(U, dx);
energy(1) = simpson1d(U.^2, dx);

%% ------------------ Time marching: n -> n+1 ------------------
for n = 1:Nt-1
    Uold = U;                                    % U^n
    [A, b] = assemble_system_1d(Uold, dx, dt);    % A(U^n), b(U^n)

    % Solve for U^{n+1}
    U = A \ b;

    % enforce boundary again (safety)
    U(1:3)   = 0;
    U(L-2:L) = 0;

    Uhist(:,n+1) = U;

    % record conservation quantities
    mass(n+1)   = simpson1d(U, dx);
    energy(n+1) = simpson1d(U.^2, dx);
end

%% ------------------ (NEW) distance between argmax and argmin vs time ------------------
xMax = zeros(Nt,1);
xMin = zeros(Nt,1);
dMaxMin = zeros(Nt,1);

% avoid boundary artifacts: search extrema only on interior points
idxInterior = 4:(L-3);

for n = 1:Nt
    Un = Uhist(:,n);

    [~, iMaxLocal] = max(Un(idxInterior));
    [~, iMinLocal] = min(Un(idxInterior));

    iMax = idxInterior(iMaxLocal);
    iMin = idxInterior(iMinLocal);

    xMax(n) = x(iMax);
    xMin(n) = x(iMin);
    dMaxMin(n) = xMax(n) - xMin(n);
end


%% ------------------ Surfaces over (x,t) ------------------
[X, Tm] = meshgrid(x, t);            % size Nt x L

% (1) Numerical solution surface
figure(1); clf;
surf(X, Tm, Uhist.');                % Uhist.' : Nt x L
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
shading interp; 
colormap(jet);
colorbar;
xlim([xL xR]);
ylim([0 T]);
xlabel('$x$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
zlabel('$u$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
%title('Numerical solution','Interpreter','latex', 'FontName','Times New Roman', 'FontSize',19);
view(0, 90);

% (2) Distance curve: |x_max(t) - x_min(t)|
figure(2); clf;
plot(t, dMaxMin, 'b-','LineWidth', 1.5); grid on;
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
xlabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$d$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
%title('Distance between max-location and min-location vs time','Interpreter','latex', 'FontName','Times New Roman', 'FontSize',19);

figure(3); clf;

t_log = log(abs(t(2:end)-t0));          % ln t
plot(t_log, (dMaxMin(2:end)), 'b-','LineWidth', 1.5); 
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
grid on;
xlabel(sprintf('$\\ln|t-%g|$', t0), ...
    'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
%xlabel('$ln|t-t_0|$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$d$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
%title('Distance between max-location and min-location vs time $\ln |t-t0|)$','Interpreter','latex', 'FontName','Times New Roman', 'FontSize',19);
%title(sprintf('Distance between max-location and min-location vs time $\\ln|t-%.4f|$', t0), ...
%    'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',19);



%% ------------------ 2D plots: mass(t) and energy(t) ------------------
figure(4); clf;
plot(t, mass, 'b-','LineWidth', 1.5); grid on;
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
%title('Mass vs time (Simpson / trapezoid fallback)');
xlabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$M$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);

figure(5); clf;
plot(t, energy, 'b-','LineWidth', 1.5); grid on;
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
xlabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$P$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
%title('Energy vs time (Simpson / trapezoid fallback)');

%% ------------------ Relative error curves: mass & energy ------------------
m0 = mass(1);
e0 = energy(1);

relMass   = abs(mass   - m0) ./ max(abs(m0), eps);
relEnergy = abs(energy - e0) ./ max(abs(e0), eps);

figure(6); clf;
plot(t, relMass, 'b-','LineWidth', 1.5); grid on;
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
xlabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$\delta M$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);

figure(7); clf;
plot(t, relEnergy, 'b-','LineWidth', 1.5); grid on;
set(gcf,'color','w');set(gca,'FontName','Times New Roman','FontSize',16);
xlabel('$t$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);
ylabel('$\delta P$', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',24);


Ufinal = Uhist(:,end);
uMaxFinal = max(Ufinal(idxInterior));
uMinFinal = min(Ufinal(idxInterior));

fprintf('Final time t=%.4f: max(U)=%.8e at x=%.8f, min(U)=%.8e at x=%.8f, d=%.8f\n', ...
    t(end), uMaxFinal, xMax(end), uMinFinal, xMin(end), dMaxMin(end));

fprintf('Relative errors at final time: relMass=%.3e, relEnergy=%.3e, peak_max=%.3e\n', ...
    relMass(end), relEnergy(end),max(max(Uhist)));

fprintf('Done.\n');
%% ============================================================
%                 Local function: assemble A and b
% ============================================================
function [A, b] = assemble_system_1d(Uold, dx, dt)
L = numel(Uold);
b = zeros(L,1);

nnz_est = 7*(L-6) + 6;
A = spalloc(L, L, nnz_est);

% boundary rows: Unew(l)=0
bd = [1,2,3, L-2, L-1, L];
for l = bd
    A(l,l) = 1.0;
    b(l)   = 0.0;
end

for l = 4:(L-3)
    ul   = Uold(l);
    um1  = Uold(l-1);
    up1  = Uold(l+1);
    um2  = Uold(l-2);
    up2  = Uold(l+2);
    um3  = Uold(l-3);
    up3  = Uold(l+3);

    du1  = (up1 - um1);
    d2   = (up1 - 2*ul + um1);

    A_lm3 = -1/(4*dx^5);
    A_lm2 = -(5/(2*dx^3))*(ul^2) + 1/(dx^5);

    A_lm1 = -(15/(2*dx))*(ul^4) ...
            -(15/(8*dx^3))*(du1^2) ...
            + (10/dx^3)*ul*( du1 - d2 ) ...
            + (5/dx^3)*(ul^2) ...
            - (5/(4*dx^5));

    A_l0  =  (1/dt) ...
            + (30/dx)*(ul^3)*du1 ...
            + (10/dx^3)*du1*( d2 - 2*ul ) ...
            + (5/dx^3)*ul*( up2 - 2*up1 + 2*um1 - um2 );

    A_lp1 =  (15/(2*dx))*(ul^4) ...
            + (15/(8*dx^3))*(du1^2) ...
            + (10/dx^3)*ul*( d2 + du1 ) ...
            - (5/dx^3)*(ul^2) ...
            + (5/(4*dx^5));

    A_lp2 =  (5/(2*dx^3))*(ul^2) - 1/(dx^5);
    A_lp3 =  1/(4*dx^5);

    A(l, l-3) = A_lm3;
    A(l, l-2) = A_lm2;
    A(l, l-1) = A_lm1;
    A(l, l  ) = A_l0;
    A(l, l+1) = A_lp1;
    A(l, l+2) = A_lp2;
    A(l, l+3) = A_lp3;

    b(l) = (1/dt)*ul ...
        + (45/(2*dx))*(ul^4)*du1 ...
        + (5/(8*dx^3))*(du1^3) ...
        + (10/dx^3)*ul*du1*d2 ...
        + (5/(2*dx^3))*(ul^2)*( up2 - 2*up1 + 2*um1 - um2 ) ...
        - (1/(4*dx^5))*( up3 - 4*up2 + 5*up1 - 5*um1 + 4*um2 - um3 );
end
end

%% ============================================================
%                 Local function: Simpson 1D
% ============================================================
function I = simpson1d(f, h)
    f = f(:);
    N = numel(f);

    if N < 2
        I = 0.0;
        return;
    end

    % if N-1 is odd, fall back to trapezoid
    if mod(N-1,2) ~= 0
        x = (0:N-1).' * h;
        I = trapz(x, f);
        return;
    end

    I = (h/3) * ( ...
        f(1) + f(N) + ...
        4*sum(f(2:2:N-1)) + ...
        2*sum(f(3:2:N-2)) );
end
