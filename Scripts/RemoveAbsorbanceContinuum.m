% Remove continuum of a Spectrum and plot

%% Patient 15 Median Spectra
% data = readtable('/Users/maycaj/Downloads/Patient15MedianSpectra');
% TrueCR = removeContinuum(data.True,data.index,Method="Division");
% FalseCR = removeContinuum(data.False,data.index,Method="Division");
% plot(data.index,TrueCR,data.index,data.True, data.index, FalseCR, data.index, data.False)
% title('Partient 15 Continuum Removed')
% legend('True Continuum removed','True','False Continuum Removed','False')

%% HbO2 and Hb
data = readtable('/Users/maycaj/Documents/HSI_III/Absorbances/Water Absorbance.csv');
data= data(76:353,:); % select range from 400nm to 954nm
HbO2 = data.HbO2Cm_1_M / max(data.HbO2Cm_1_M);
Hb = data.HbCm_1_M / max(data.HbCm_1_M);
HbO2CR = removeContinuum(data.HbCm_1_M, data.lambdaNm);
HbCR = removeContinuum(data.HbO2Cm_1_M, data.lambdaNm);
plot(data.lambdaNm, HbO2, data.lambdaNm, Hb, data.lambdaNm, HbO2CR, data.lambdaNm, HbCR);
legend('HbO2','Hb', 'HbO2 CR','Hb CR')
