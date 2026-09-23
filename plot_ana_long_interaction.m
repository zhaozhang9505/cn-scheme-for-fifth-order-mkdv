% Plot Eq. (4.3) for three epsilon values and Eq. (4.4) in a 2-by-2 layout.

clear; clc; close all;

%% ------------------ Parameters ------------------
k = 1;
beta = 1;
epsilonVals = [1/4, 1/8, 1/20];

nx = 601;
nt = 501;
x = linspace(-20, 20, nx);
t = linspace(-10, 10, nt);
[X, T] = meshgrid(x, t);

Z = cell(1, 4);
zMin = inf;
zMax = -inf;

for n = 1:numel(epsilonVals)
    Z{n} = eq43_solution(X, T, k, beta, epsilonVals(n));
    finiteMask = isfinite(Z{n});
    zMin = min(zMin, min(Z{n}(finiteMask), [], "all"));
    zMax = max(zMax, max(Z{n}(finiteMask), [], "all"));
end

Z{4} = eq44_solution(X, T, k, beta);
finiteMask = isfinite(Z{4});
zMin = min(zMin, min(Z{4}(finiteMask), [], "all"));
zMax = max(zMax, max(Z{4}(finiteMask), [], "all"));

%% ------------------ 2-by-2 top-view surfaces ------------------
fig = figure(1); clf;
set(fig, 'Color', 'w', 'Name', 'Analytical prolonged interaction');

tl = tiledlayout(fig, 2, 2, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

panelTitles = { ...
    '$\varepsilon=1/4$', ...
    '$\varepsilon=1/8$', ...
    '$\varepsilon=1/20$', ...
    '$\varepsilon=0$'};

for n = 1:4
    ax = nexttile(tl);
    surf(ax, X, T, Z{n}, 'EdgeColor', 'none');
    shading(ax, 'interp');
    view(ax, 0, 90);
    axis(ax, 'tight');
    xlim(ax, [x(1), x(end)]);
    ylim(ax, [t(1), t(end)]);
    clim(ax, [zMin, zMax]);
    box(ax, 'on');

    set(ax, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 16);
    xlabel(ax, '$x$', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);
    ylabel(ax, '$t$', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);
    title(ax, panelTitles{n}, ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 20);
end

colormap(fig, jet);
cb = colorbar;
cb.Layout.Tile = 'east';
%cb.Label.String = '$u$';
cb.Label.Interpreter = 'latex';
cb.FontName = 'Times New Roman';
cb.FontSize = 16;
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = 24;

%% ============================================================
%                 Local function: Eq. (4.3)
% ============================================================
function U = eq43_solution(x, t, k, beta, epsilon)
xiP = (k + epsilon) .* (x - (k + epsilon)^4 .* t);
xiM = (k - epsilon) .* (x - (k - epsilon)^4 .* t);
xi0 = 2*k .* (x - (k^4 + 10*epsilon^2*k^2 + 5*epsilon^4) .* t);

num = 2*epsilon*beta*k^2 .* ( ...
    beta^2 .* exp(xi0) .* ( ...
        (k + epsilon).*exp(xiM) - (k - epsilon).*exp(xiP) ...
    ) ...
    + k^2 .* ( ...
        (epsilon - k).*exp(xiM) + (k + epsilon).*exp(xiP) ...
    ) ...
);

den = beta^4*epsilon^2 .* exp(2*xi0) ...
    + 2*beta^2*k^2*(epsilon^2 - k^2) .* exp(xi0) ...
    + k^4 .* ( ...
        beta^2 .* (exp(2*xiP) + exp(2*xiM)) + epsilon^2 ...
    );

U = num ./ den;
end

%% ============================================================
%                 Local function: Eq. (4.4)
% ============================================================
function U = eq44_solution(x, t, k, beta)
theta = k .* (x - k^4 .* t);
s = 5*k^5 .* t - k .* x;
E = exp(theta);
E2 = exp(2*theta);

num = 4*k^2*beta .* E .* ( ...
    s .* (beta^2 .* E2 - k^2) + beta^2 .* E2 + k^2 ...
);

den = (beta^2 .* E2 + k^2).^2 ...
    + 4*k^2 .* s.^2 .* beta^2 .* E2;

U = num ./ den;
end
