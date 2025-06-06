data = readtable('/Users/maycaj/Documents/HSI_III/1-22-25_5x5.csv'); 

%%
clf;

rowIDX = randi([0,height(data)-1]);

spectra = data{rowIDX, 2:129};

% spectra = [1 2 4 2 1 2 3 2 1];

wavelength = 1:length(spectra); % Assuming evenly spaced wavelengths

% Get convex hull indices
[convIDX, av] = convhull(wavelength, spectra);

convHull = spectra(convIDX);
convWavelengths = wavelength(convIDX);

spectra_Cr_orig = removeContinuum(spectra, wavelength);

% Plot everything
figure1 = figure(1);
hold on
plot(wavelength, spectra, 'r', 'LineWidth', 1.5)  % Original spectrum
plot(convWavelengths, convHull, 'g') % upper points
% plot(wavelength, continuum, 'g--', 'LineWidth', 1.5) % Continuum
% plot(wavelength, spectra_Cr, 'r', 'LineWidth', 1.5) % Continuum-removed spectrum
plot(wavelength, spectra_Cr_orig, 'bl', 'LineWidth',1.5)
legend('Original Spectra', 'Convex Hull', 'removeContinuum() on spectra')
xlabel('Wavelength')
ylabel('Reflectance')
title('Continuum Removal')
grid on
hold off

