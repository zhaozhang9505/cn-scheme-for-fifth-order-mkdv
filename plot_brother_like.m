% Plot Eq. (5.2) for three epsilon values and keep Eq. (4.4)
% in the fourth panel.

clear; clc; close all;

%% ------------------ Parameters ------------------
k = 1;
beta = 1;
epsilonVals = [1/4, 1/8, 1/25];

nx = 601;
nt = 501;
x = linspace(-20, 20, nx);
t = linspace(-10, 10, nt);
[X, T] = meshgrid(x, t);

zRange = [-2, 2];

Z = cell(1, 4);

for n = 1:numel(epsilonVals)
    Z{n} = eq52_solution(X, T, k, beta, epsilonVals(n));
end

Z{4} = eq44_solution(X, T, k, beta);

%% ------------------ 2-by-2 top-view surfaces ------------------
fig = figure(1); clf;
set(fig, 'Color', 'w', 'Name', 'Plot brother-like');

tl = tiledlayout(fig, 2, 2, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

panelTitles = { ...
    '$\varepsilon=1/4$', ...
    '$\varepsilon=1/8$', ...
    '$\varepsilon=1/25$', ...
    '$\varepsilon=0$'};

for n = 1:4
    ax = nexttile(tl);
    surf(ax, X, T, Z{n}, 'EdgeColor', 'none');
    shading(ax, 'interp');
    view(ax, 0, 90);
    axis(ax, 'tight');
    xlim(ax, [x(1), x(end)]);
    ylim(ax, [t(1), t(end)]);
    zlim(ax, zRange);
    clim(ax, zRange);
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
%                 Local function: Eq. (5.2)
% ============================================================
function U = eq52_solution(x, t, k, beta, epsilon)
rho = (beta/k) .* exp(k .* ( ...
    x - (k^4 - 10*k^2*epsilon^2 + 5*epsilon^4) .* t ...
));

Phi = epsilon .* ( ...
    x - (5*k^4 - 10*k^2*epsilon^2 + epsilon^4) .* t ...
);

num = 4*k .* rho .* ( ...
    (1 + rho.^2) .* cos(Phi) ...
    + k .* (1 - rho.^2) ./ epsilon .* sin(Phi) ...
);

den = (1 + rho.^2).^2 ...
    + 4*k^2 .* rho.^2 ./ epsilon^2 .* sin(Phi).^2;

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
