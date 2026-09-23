% Plot the real part of the supplied expression for four epsilon values.

clear; clc; close all;

k = 1;
beta = 1;
epsilonVals = [1/4, 1/8, 1/12, 1/24];
plotQuantity = "real";

nx = 401;
nt = 401;
x = linspace(-30, 30, nx);
t = linspace(-20, 20, nt);
[X, T] = meshgrid(x, t);

Z = cell(size(epsilonVals));
zMin = inf;
zMax = -inf;

for n = 1:numel(epsilonVals)
    U = suppliedExpression(X, T, k, beta, epsilonVals(n));

    switch lower(plotQuantity)
        case "abs"
            Zn = abs(U);
        case "real"
            Zn = real(U);
        case "imag"
            Zn = imag(U);
        otherwise
            error('plotQuantity must be "abs", "real", or "imag".');
    end

    Z{n} = Zn;
    finiteMask = isfinite(Zn);
    zMin = min(zMin, min(Zn(finiteMask), [], "all"));
    zMax = max(zMax, max(Zn(finiteMask), [], "all"));
end

fig = figure("Color", "w", "Name", "Expression top views");
tl = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");

for n = 1:numel(epsilonVals)
    ax = nexttile(tl);
    surf(ax, X, T, Z{n}, "EdgeColor", "none");
    view(ax, 2);
    axis(ax, "tight");
    shading(ax, "interp");
    clim(ax, [zMin, zMax]);
    ax.FontName = "Times New Roman";
    ax.FontSize = 15;
    xlabel(ax, "\it x", "FontName", "Times New Roman", "FontSize", 20);
    ylabel(ax, "\it t", "FontName", "Times New Roman", "FontSize", 20);
    title(ax, sprintf("\\epsilon = 1/%g", round(1/epsilonVals(n))), ...
        "FontName", "Times New Roman", "FontSize", 20);
    box(ax, "on");
end

colormap(fig, coolToWarm(256));
cb = colorbar;
cb.Layout.Tile = "east";
cb.Label.String = "\it u";
cb.FontName = "Times New Roman";
cb.FontSize = 15;
cb.Label.FontName = "Times New Roman";
cb.Label.FontSize = 20;

function U = suppliedExpression(x, t, k, beta, epsilon)
I = 1i;
A = k^2 + (1/4)*epsilon^2;

N1 = -1/2*A^2*(I*k*epsilon - k^2 + (1/4)*epsilon^2)*k^4*beta^2 .* ...
    exp((-3*k^5 + 20*epsilon^2*k^3 - 10*I*k^4*epsilon - 10*k*epsilon^4 - 2*I*epsilon^5 + 20*I*k^2*epsilon^3).*t + 2*(I*epsilon + (3/2)*k).*x);

N2 = 1/2*A^2*(I*k*epsilon + k^2 - (1/4)*epsilon^2)*k^4*beta^2 .* ...
    exp((-3*k^5 + 20*epsilon^2*k^3 + 10*I*k^4*epsilon - 10*k*epsilon^4 + 2*I*epsilon^5 - 20*I*k^2*epsilon^3).*t - 2*(I*epsilon - (3/2)*k).*x);

N3 = A^4*(I*epsilon - k)*k*beta^2 .* ...
    exp((-15*k*epsilon^4 + 30*epsilon^2*k^3 - 3*k^5 - 5*I*k^4*epsilon + 10*I*k^2*epsilon^3 - I*epsilon^5).*t + (I*epsilon + 3*k).*x);

N4 = 1/4*A^2*(I*epsilon^3 - 4*k^3 - 3*k*epsilon^2)*k^3*beta^2 .* ...
    exp((-3*k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (I*epsilon + 3*k).*x);

N5 = -1/8*(I*k^2*epsilon - (1/8)*I*epsilon^3 - (1/2)*k^3 + (5/8)*k*epsilon^2)*epsilon^2*k*beta^4 .* ...
    exp((-15*k*epsilon^4 + 30*epsilon^2*k^3 - 5*k^5 - 5*I*k^4*epsilon + 10*I*k^2*epsilon^3 - I*epsilon^5).*t + (I*epsilon + 5*k).*x);

N6 = -1/4*A^2*k^3*(I*epsilon^3 + 4*k^3 + 3*k*epsilon^2)*beta^2 .* ...
    exp((-3*k^5 + 5*I*k^4*epsilon + 10*epsilon^2*k^3 - 10*I*k^2*epsilon^3 - 5*k*epsilon^4 + I*epsilon^5).*t - (I*epsilon - 3*k).*x);

N7 = A^4*epsilon^2*k^3*(k + I*epsilon) .* ...
    exp((-k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (k + I*epsilon).*x);

N8 = -A^4*k*(k + I*epsilon)*beta^2 .* ...
    exp(I.*t*epsilon^5 - 15.*t*k*epsilon^4 - 10*I.*t*k^2*epsilon^3 + 30.*t*k^3*epsilon^2 + I*(5*k^4.*t - x)*epsilon - 3*k^5.*t + 3*k.*x);

N9 = 1/8*epsilon^2*k*(I*k^2*epsilon - (1/8)*I*epsilon^3 + (1/2)*k^3 - (5/8)*k*epsilon^2)*beta^4 .* ...
    exp(I.*t*epsilon^5 - 15.*t*k*epsilon^4 - 10*I.*t*k^2*epsilon^3 + 30.*t*k^3*epsilon^2 + I*(5*k^4.*t - x)*epsilon - 5*k^5.*t + 5*k.*x);

N10a = A^2*(I*epsilon - k)*epsilon^2*k^3 .* ...
    exp(I.*t*epsilon^5 - 5.*t*k*epsilon^4 - 10*I.*t*k^2*epsilon^3 + 10.*t*k^3*epsilon^2 + I*(5*k^4.*t - x)*epsilon - k^5.*t + k.*x);

N10b = 1/8*beta^4*epsilon^2 .* ...
    exp(-5*((4*epsilon^4 - 8*epsilon^2*k^2 + k^4).*t - x)*k);

N10c = 2*(-3/2*(k^2 - (1/12)*epsilon^2)*(epsilon^2 + k^2)*beta^2 .* ...
    exp(-3*k^5.*t + 20.*t*k^3*epsilon^2 + (-10*epsilon^4.*t + 3.*x)*k) + ...
    A^2*epsilon^2*k^2 .* exp(-k*(k^4.*t - x))) * k^2;

N10 = -A^2 * (N10a + N10b + N10c);
Numerator = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10;

D1 = -1/2*(I*k^2 - (1/4)*I*epsilon^2 + k*epsilon)*k^2*beta^2 .* ...
    exp((-2*k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (2*k + I*epsilon).*x) ...
    - 1/2*k^2*(I*k^2 - (1/4)*I*epsilon^2 - k*epsilon)*beta^2 .* ...
    exp((-2*k^5 + 5*I*k^4*epsilon + 10*epsilon^2*k^3 - 10*I*k^2*epsilon^3 - 5*k*epsilon^4 + I*epsilon^5).*t - (-2*k + I*epsilon).*x) ...
    + A^2*k^2*beta .* ...
    exp((-k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (k + I*epsilon).*x) ...
    + A^2*k^2*beta .* ...
    exp(I.*t*epsilon^5 - 5.*t*k*epsilon^4 - 10*I.*t*k^2*epsilon^3 + 10.*t*k^3*epsilon^2 + I*(5*k^4.*t - x)*epsilon - k^5.*t + k.*x) ...
    + I*A^2*beta^2 .* exp(-2*((5*epsilon^4 - 10*epsilon^2*k^2 + k^4).*t - x)*k) ...
    - 1/8*beta^3*epsilon^2 .* exp(-3*k^5.*t + 20.*t*k^3*epsilon^2 + (-10*epsilon^4.*t + 3.*x)*k) ...
    + A^2*k^2 .* (I*epsilon^2 - 2*beta .* exp(-k*(k^4.*t - x)));

D2 = -1/2*(I*k^2 - (1/4)*I*epsilon^2 + k*epsilon)*k^2*beta^2 .* ...
    exp((-2*k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (2*k + I*epsilon).*x) ...
    - 1/2*k^2*(I*k^2 - (1/4)*I*epsilon^2 - k*epsilon)*beta^2 .* ...
    exp((-2*k^5 + 5*I*k^4*epsilon + 10*epsilon^2*k^3 - 10*I*k^2*epsilon^3 - 5*k*epsilon^4 + I*epsilon^5).*t - (-2*k + I*epsilon).*x) ...
    - A^2*k^2*beta .* ...
    exp((-k^5 - 5*I*k^4*epsilon + 10*epsilon^2*k^3 + 10*I*k^2*epsilon^3 - 5*k*epsilon^4 - I*epsilon^5).*t + (k + I*epsilon).*x) ...
    - A^2*k^2*beta .* ...
    exp(I.*t*epsilon^5 - 5.*t*k*epsilon^4 - 10*I.*t*k^2*epsilon^3 + 10.*t*k^3*epsilon^2 + I*(5*k^4.*t - x)*epsilon - k^5.*t + k.*x) ...
    + I*A^2*beta^2 .* exp(-2*((5*epsilon^4 - 10*epsilon^2*k^2 + k^4).*t - x)*k) ...
    + 1/8*beta^3*epsilon^2 .* exp(-3*k^5.*t + 20.*t*k^3*epsilon^2 + (-10*epsilon^4.*t + 3.*x)*k) ...
    + A^2*k^2 .* (I*epsilon^2 + 2*beta .* exp(-k*(k^4.*t - x)));

U = 2*k*beta .* Numerator ./ D1 ./ D2;
end

function cmap = coolToWarm(n)
anchors = [ ...
    0.0500 0.1800 0.5200
    0.0800 0.4300 0.7800
    0.5200 0.7800 0.9300
    0.9500 0.9500 0.9000
    0.9600 0.6200 0.2600
    0.7600 0.0800 0.1200];
xi = linspace(0, 1, size(anchors, 1));
xq = linspace(0, 1, n);
cmap = interp1(xi, anchors, xq, "pchip");
cmap = max(0, min(1, cmap));
end
