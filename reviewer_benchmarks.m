function reviewer_benchmarks(mode)
% Reproducible one-soliton tests for the Wave Motion revision.
% Uses the seven-diagonal coefficients in Sec_3_1.m, but measures error
% against the UNMODIFIED exact soliton at every grid node.
% Run from this folder: matlab -batch reviewer_benchmarks

repetitions = 3;
T = 1;
outdir = fileparts(mfilename('fullpath'));
if nargin == 0
    mode = 'standard';
end
reported_repeats = repetitions;
if strcmp(mode, 'regression_all') || strcmp(mode, 'regression_long')
    reported_repeats = 1;
end
fprintf('MATLAB %s (%s)\n', version, computer('arch'));
fprintf('Processor: %s\n', getenv('PROCESSOR_IDENTIFIER'));
fprintf('Repetitions per case: %d; standard final time: %.3f\n', reported_repeats, T);

% Warm up the JIT and sparse linear solver; this run is not timed.
run_case(-8, 8, 0.2, 0.1, 0.1);

if strcmp(mode, 'fine')
    dt_fine = [0.1; 0.05; 0.025; 0.0125; 0.00625];
    fine = run_series(repmat(-20, 5, 1), repmat(20, 5, 1), ...
        repmat(0.00625, 5, 1), dt_fine, repmat(T, 5, 1), repetitions);
    fine.order_inf = [NaN; log2(fine.E_inf(1:end-1)./fine.E_inf(2:end))];
    fine.order_2 = [NaN; log2(fine.E_2(1:end-1)./fine.E_2(2:end))];
    writetable(fine, fullfile(outdir, 'reviewer_temporal_fine.csv'));
    disp('Additional finer spatial grid:'); disp(fine);
    return;
end
if strcmp(mode, 'ultrafine')
    % Halving h once more tests whether the temporal rates are affected by
    % the fixed-grid spatial error. Keep the same exact wave and final time.
    dt_ultrafine = [0.1; 0.05; 0.025; 0.0125];
    ultrafine = run_series(repmat(-20, 4, 1), repmat(20, 4, 1), ...
        repmat(0.003125, 4, 1), dt_ultrafine, repmat(T, 4, 1), repetitions);
    ultrafine.order_inf = [NaN; log2(ultrafine.E_inf(1:end-1)./ultrafine.E_inf(2:end))];
    ultrafine.order_2 = [NaN; log2(ultrafine.E_2(1:end-1)./ultrafine.E_2(2:end))];
    writetable(ultrafine, fullfile(outdir, 'reviewer_temporal_ultrafine.csv'));
    disp('Temporal refinement on the ultrafine spatial grid:'); disp(ultrafine);
    return;
end
if strcmp(mode, 'ultrafine_extra')
    % Compute only the missing Table 2 level; preserve the four-row CSV.
    previous = readtable(fullfile(outdir, 'reviewer_temporal_ultrafine.csv'));
    idx = find(abs(previous.h-0.003125) < 1e-12 & ...
        abs(previous.dt-0.0125) < 1e-12 & abs(previous.T-T) < 1e-12);
    assert(numel(idx) == 1, 'Expected one matching reference row.');
    extra = run_series(-20, 20, 0.003125, 0.00625, T, repetitions);
    extra.order_inf = log2(previous.E_inf(idx)/extra.E_inf(1));
    extra.order_2 = log2(previous.E_2(idx)/extra.E_2(1));
    writetable(extra, fullfile(outdir, 'reviewer_temporal_ultrafine_extra.csv'));
    disp('Additional Table 2 temporal level:'); disp(extra);
    return;
end
if strcmp(mode, 'mid')
    dt_mid = [0.1; 0.05; 0.025; 0.0125; 0.00625];
    mid = run_series(repmat(-20, 5, 1), repmat(20, 5, 1), ...
        repmat(0.025, 5, 1), dt_mid, repmat(T, 5, 1), repetitions);
    mid.order_inf = [NaN; log2(mid.E_inf(1:end-1)./mid.E_inf(2:end))];
    mid.order_2 = [NaN; log2(mid.E_2(1:end-1)./mid.E_2(2:end))];
    writetable(mid, fullfile(outdir, 'reviewer_temporal_mid.csv'));
    disp('Intermediate spatial grid:'); disp(mid);
    return;
end
if strcmp(mode, 'spatial')
    spatial_h = [0.2; 0.1; 0.05; 0.025];
    spatial = run_series(repmat(-20, 4, 1), repmat(20, 4, 1), ...
        spatial_h, repmat(0.005, 4, 1), repmat(T, 4, 1), repetitions);
    spatial.order_inf = [NaN; log2(spatial.E_inf(1:end-1)./spatial.E_inf(2:end))];
    spatial.order_2 = [NaN; log2(spatial.E_2(1:end-1)./spatial.E_2(2:end))];
    writetable(spatial, fullfile(outdir, 'reviewer_spatial.csv'));
    disp('Independent spatial refinement:'); disp(spatial);
    return;
end
if strcmp(mode, 'regression_all')
    hh = [0.2; 0.1; 0.1; 0.05; 0.05; 0.025];
    dd = [0.1; 0.1; 0.05; 0.05; 0.025; 0.025];
    old = run_series(repmat(-20, 6, 1), repmat(20, 6, 1), ...
        hh, dd, repmat(2, 6, 1), 1);
    writetable(old, fullfile(outdir, 'reviewer_regression_all.csv'));
    disp('Original Table 1 settings, reduced solve:'); disp(old);
    return;
end
if strcmp(mode, 'regression_long')
    long = run_series(-20, 60, 0.025, 0.025, 40, 1);
    writetable(long, fullfile(outdir, 'reviewer_regression_long.csv'));
    disp('Original Table 2 case 6 setting, reduced solve:'); disp(long);
    return;
end

temporal_dt = [0.1; 0.05; 0.025; 0.0125];
temporal_h = 0.0125;
temporal = run_series(repmat(-20, 4, 1), repmat(20, 4, 1), ...
    repmat(temporal_h, 4, 1), temporal_dt, repmat(T, 4, 1), repetitions);
temporal.order_inf = [NaN; log2(temporal.E_inf(1:end-1)./temporal.E_inf(2:end))];
temporal.order_2 = [NaN; log2(temporal.E_2(1:end-1)./temporal.E_2(2:end))];
writetable(temporal, fullfile(outdir, 'reviewer_temporal.csv'));

spatial_h = [0.2; 0.1; 0.05; 0.025];
spatial_dt = 0.005;
spatial = run_series(repmat(-20, 4, 1), repmat(20, 4, 1), ...
    spatial_h, repmat(spatial_dt, 4, 1), repmat(T, 4, 1), repetitions);
spatial.order_inf = [NaN; log2(spatial.E_inf(1:end-1)./spatial.E_inf(2:end))];
spatial.order_2 = [NaN; log2(spatial.E_2(1:end-1)./spatial.E_2(2:end))];
writetable(spatial, fullfile(outdir, 'reviewer_spatial.csv'));

domain_a = [-8; -12; -20];
domain_b = [8; 12; 20];
domains = run_series(domain_a, domain_b, repmat(0.05, 3, 1), ...
    repmat(0.01, 3, 1), repmat(T, 3, 1), repetitions);
writetable(domains, fullfile(outdir, 'reviewer_domain.csv'));

% A published short-time Table 1 setting for a regression check.
regression = run_series(-20, 20, 0.2, 0.1, 2, repetitions);
writetable(regression, fullfile(outdir, 'reviewer_regression.csv'));

disp('Temporal refinement:'); disp(temporal);
disp('Spatial refinement:'); disp(spatial);
disp('Domain sensitivity:'); disp(domains);
disp('Original Table 1 case 1 regression:'); disp(regression);
end

function result = run_series(xL, xR, h, dt, T, repeats)
ncase = numel(h);
E_inf = zeros(ncase, 1); E_2 = zeros(ncase, 1);
wall_s = zeros(ncase, 1); cpu_s = zeros(ncase, 1);
assembly_s = zeros(ncase, 1); solve_s = zeros(ncase, 1);
N_x = zeros(ncase, 1); N_t = zeros(ncase, 1);
for q = 1:ncase
    timing = zeros(repeats, 4);
    for rep = 1:repeats
        [err_inf, err_2, timing(rep,:), nx, nt] = ...
            run_case(xL(q), xR(q), h(q), dt(q), T(q));
        if rep == 1
            E_inf(q) = err_inf; E_2(q) = err_2;
            N_x(q) = nx; N_t(q) = nt;
        elseif abs(err_inf-E_inf(q)) > 1e-10 || abs(err_2-E_2(q)) > 1e-10
            error('Non-repeatable numerical result at case %d.', q);
        end
    end
    med = median(timing, 1);
    wall_s(q) = med(1); cpu_s(q) = med(2);
    assembly_s(q) = med(3); solve_s(q) = med(4);
    fprintf('case %d: h=%.6g dt=%.6g [%g,%g] T=%g  E_inf=%.6e E_2=%.6e wall=%.3fs CPU=%.3fs\n', ...
        q, h(q), dt(q), xL(q), xR(q), T(q), E_inf(q), E_2(q), wall_s(q), cpu_s(q));
end
result = table(xL, xR, h, dt, T, N_x, N_t, E_inf, E_2, ...
    wall_s, cpu_s, assembly_s, solve_s);
end

function [E_inf, E_2, times, N_x, N_t] = run_case(xL, xR, h, dt, T)
assert(abs(round((xR-xL)/h)-(xR-xL)/h) < 1e-9);
assert(abs(round(T/dt)-T/dt) < 1e-9);
x = (xL:h:xR).';
N_x = numel(x); N_t = round(T/dt)+1;
assert(N_x >= 7);
u = sech(x);
u([1:3, N_x-2:N_x]) = 0;
assembly_time = 0; solve_time = 0;
c0 = cputime; w0 = tic;
for n = 1:N_t-1
    a0 = tic;
    [A,b] = assemble_original(u, h, dt);
    assembly_time = assembly_time + toc(a0);
    s0 = tic;
    % Eliminate the six fixed zero nodes before the sparse solve.
    % This is algebraically equivalent to the legacy full-system solve,
    % but avoids mixing identity boundary rows with h^(-5) interior rows.
    interior = 4:N_x-3;
    u(interior) = A(interior,interior)\b(interior);
    solve_time = solve_time + toc(s0);
    u([1:3, N_x-2:N_x]) = 0;
    if any(~isfinite(u))
        error('Nonfinite numerical solution at step %d.', n);
    end
end
times = [toc(w0), cputime-c0, assembly_time, solve_time];
exact = sech(x-T);
err = u-exact;
E_inf = max(abs(err));
E_2 = sqrt(h*sum(err.^2));
end

function [A,b] = assemble_original(Uold, dx, dt)
L = numel(Uold);
b = zeros(L,1);
A = spalloc(L,L,7*(L-6)+6);
bd = [1,2,3,L-2,L-1,L];
for l = bd
    A(l,l) = 1;
end
for l = 4:L-3
    ul = Uold(l);
    um1 = Uold(l-1); up1 = Uold(l+1);
    um2 = Uold(l-2); up2 = Uold(l+2);
    um3 = Uold(l-3); up3 = Uold(l+3);
    du1 = up1-um1;
    d2 = up1-2*ul+um1;
    A(l,l-3) = -1/(4*dx^5);
    A(l,l-2) = -(5/(2*dx^3))*ul^2 + 1/dx^5;
    A(l,l-1) = -(15/(2*dx))*ul^4 -(15/(8*dx^3))*du1^2 ...
        +(10/dx^3)*ul*(du1-d2)+(5/dx^3)*ul^2-5/(4*dx^5);
    A(l,l) = 1/dt+(30/dx)*ul^3*du1 ...
        +(10/dx^3)*du1*(d2-2*ul) ...
        +(5/dx^3)*ul*(up2-2*up1+2*um1-um2);
    A(l,l+1) = (15/(2*dx))*ul^4+(15/(8*dx^3))*du1^2 ...
        +(10/dx^3)*ul*(d2+du1)-(5/dx^3)*ul^2+5/(4*dx^5);
    A(l,l+2) = (5/(2*dx^3))*ul^2-1/dx^5;
    A(l,l+3) = 1/(4*dx^5);
    b(l) = ul/dt+(45/(2*dx))*ul^4*du1+(5/(8*dx^3))*du1^3 ...
        +(10/dx^3)*ul*du1*d2 ...
        +(5/(2*dx^3))*ul^2*(up2-2*up1+2*um1-um2) ...
        -(1/(4*dx^5))*(up3-4*up2+5*up1-5*um1+4*um2-um3);
end
end
