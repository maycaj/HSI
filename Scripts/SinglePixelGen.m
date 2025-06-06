% Define folder paths and their respective final strings
folder1Path = '/Users/maycaj/Documents/HSI_III/EdemaTrueCropsMat';
folder2Path = '/Users/maycaj/Documents/HSI_III/EdemaFalseCropsMat';
finalString1 = 'True';
finalString2 = 'False';

% Temporary CSV file to store all data line by line
tempCSV = 'Feb4_1x1_ContinuumRemoved.csv';

% Initialize column headers
wavelengths = [376.61, 381.55, 386.49, 391.43, 396.39, 401.34, 406.30, 411.27, 416.24, ...
               421.22, 426.20, 431.19, 436.18, 441.17, 446.17, 451.18, 456.19, 461.21, ...
               466.23, 471.25, 476.28, 481.32, 486.36, 491.41, 496.46, 501.51, 506.57, ...
               511.64, 516.71, 521.78, 526.86, 531.95, 537.04, 542.13, 547.23, 552.34, ...
               557.45, 562.57, 567.69, 572.81, 577.94, 583.07, 588.21, 593.36, 598.51, ...
               603.66, 608.82, 613.99, 619.16, 624.33, 629.51, 634.70, 639.88, 645.08, ...
               650.28, 655.48, 660.69, 665.91, 671.12, 676.35, 681.58, 686.81, 692.05, ...
               697.29, 702.54, 707.80, 713.06, 718.32, 723.59, 728.86, 734.14, 739.42, ...
               744.71, 750.01, 755.30, 760.61, 765.92, 771.23, 776.55, 781.87, 787.20, ...
               792.53, 797.87, 803.21, 808.56, 813.91, 819.27, 824.63, 830.00, 835.37, ...
               840.75, 846.13, 851.52, 856.91, 862.31, 867.71, 873.12, 878.53, 883.95, ...
               889.37, 894.80, 900.23, 905.67, 911.11, 916.56, 922.01, 927.47, 932.93, ...
               938.40, 943.87, 949.35, 954.83, 960.31, 965.81, 971.30, 976.80, 982.31, ...
               987.82, 993.34, 998.86, 1004.39, 1009.92, 1015.45, 1020.99, 1026.54, ...
               1032.09, 1037.65, 1043.21];
headers = ['PatientNumber,ImageNumber,', ...
           strjoin(arrayfun(@(x) sprintf('Wavelength_%.2f', x), wavelengths, 'UniformOutput', false), ','), ...
           ',FinalString'];

% Open the CSV file for writing
fileID = fopen(tempCSV, 'w');
fprintf(fileID, '%s\n', headers);
fclose(fileID);

% Process both folders
folders = {folder1Path, folder2Path};
finalStrings = {finalString1, finalString2};

for fIdx = 1:2
    folderPath = folders{fIdx};
    finalString = finalStrings{fIdx};
    
    disp(['Processing folder: ', folderPath]);
    matFiles = dir(fullfile(folderPath, '*_hypercube.mat'));
    
    for i = 1:length(matFiles) % Process each mat file in the folder
        try
            % File path and name
            filePath = fullfile(matFiles(i).folder, matFiles(i).name);
            fileName = matFiles(i).name;
            disp(['Processing file: ', fileName]);

            % Load .mat file
            matData = load(filePath);
            if ~isfield(matData, 'hc')
                disp(['Error: Hypercube not found in file: ', fileName]);
                continue;
            end

            % Extract hypercube
            hcubeunremove = matData.hc;
            hypercube = removeContinuum(hcubeunremove);
            hypercube = hypercube.DataCube;
            [rows, cols, wavelengths] = size(hypercube);

            if wavelengths ~= 128
                error(['Hypercube does not have 128 spectral dimensions in file: ', fileName]);
            end

            % Extract patient and image numbers
            patientNumber = str2double(regexp(fileName, '(?<=Edema\s)\d+', 'match', 'once'));
            imageNumber = str2double(regexp(fileName, '(?i)(?<=image\s)\d+', 'match', 'once'));

            % Handle invalid patient/image numbers
            if isnan(patientNumber)
                disp(['Error: Invalid patient number in file: ', fileName]);
                patientNumber = input('Enter patient number: ');
            end

            if isempty(imageNumber)
                disp(['Error: Invalid image number in file: ', fileName]);
                imageNumber = input('Enter image number: ');
            end

            % Open the CSV file for appending
            fileID = fopen(tempCSV, 'a');

            % Process hypercube
            for x = 1:rows
                for y = 1:cols
                    spectralVector = reshape(hypercube(x, y, :), [1, 128]);

                    % Skip invalid or zero vectors
                    if all(spectralVector == 1) || any(isnan(spectralVector))
                        continue;
                    end

                    % Write to CSV
                    fprintf(fileID, '%d,%d,', patientNumber, imageNumber);
                    fprintf(fileID, '%.6f,', spectralVector);
                    fprintf(fileID, '%s\n', finalString);
                end
            end

            fclose(fileID);

            disp(['Successfully processed file: ', fileName]);
        catch ME
            disp(['Error processing file: ', fileName]);
            disp(['Error message: ', ME.message]);
            continue;
        end
    end
end

% Read the temporary CSV back into MATLAB
%data = readtable(tempCSV);

% Get unique patient numbers
%uniquePatients = unique(data.PatientNumber);
%numPatients = numel(uniquePatients);
%trainSize = round(0.8 * numPatients);

% Shuffle patient numbers for splitting
%rng(42); % Set seed for reproducibility
%shuffledPatients = uniquePatients(randperm(numPatients));

% Split patient numbers into training and testing sets
%trainPatients = shuffledPatients(1:trainSize);
%testPatients = shuffledPatients(trainSize+1:end);

% Split data into training and testing sets
%trainData = data(ismember(data.PatientNumber, trainPatients), :);
%testData = data(ismember(data.PatientNumber, testPatients), :);

% Write training and testing data to separate CSV files
%writetable(trainData, 'train_data.csv');
%writetable(testData, 'test_data.csv');

%disp('Script finished successfully!');