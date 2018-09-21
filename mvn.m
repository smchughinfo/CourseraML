# https://octave.sourceforge.io/statistics/function/mvnpdf.html
# Multivariate Normal Probability Density Function
# this is just to see, hands on, what he's doing in wek9 video7

function mvn(mu, sigma)
    if(exist("mu", "var") == 0)
        fprintf('\nsuggested value for mu is [0 0]');
        fprintf('\nsuggested value for Sigma is [1 .1;.1 .5]\n');
        return;
    endif

    [X, Y] = meshgrid (linspace (-3, 3, 25));
    XY = [X(:), Y(:)];
    Z = mvnpdf (XY, mu, sigma);
    mesh (X, Y, reshape (Z, size (X)));
    colormap jet
end