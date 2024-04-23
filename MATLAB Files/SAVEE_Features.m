% Features extracted based on SAVEE Dataset

% Specify the directory containing your WAV files
dataset_directory = "C:\Users\User\Downloads\All Datasets\SAVEE_Dataset";

% Get a list of all WAV files in the directory
wav_files = dir(fullfile(dataset_directory, '*.wav'));

WindowSize = 2048;
Overlap = 1024;

order = 12;     % Used to define the order of LPC

% Initialize a matrix to store the features
all_statistics = zeros(length(wav_files), 120); 
            % 13 MFCC coefficients + 1 Mean MFCC + 1 Std MFCC
            % 13 Delta MFCC coefficients + 1 Mean Delta MFCC + 1 Std Delta MFCC
            % 13 DeltaDelta MFCC coefficients + 1 Mean DeltaDelta MFCC + 1 Std DeltaDelta MFCC
            % 32 Mel Spectrogram + 1 Mean  Mel Spectrogram + 1 Std Mel Spectrogram
            % 1 Spectral Flux
            % 1 Spectral Skewness
            % 1 Spectral Slope
            % 1 Spectral Entropy
            % 1 Spectral Rolloff
            % 12 Chromagram coefficients + 1 Mean Chromagram + 1 Std Chromagram
            % 12 LPC coefficients + 1 Mean LPC + 1 Std LPC
            % 1 ZCR
            % 1 Total Energy + 1 Mean Energy + 1 Std Energy
            % 1 Pitch
            % 1 Intensity
            % 1 HNR
            % 1 RMS

% Loop through each WAV file in the dataset
for i = 1:length(wav_files)
    file_name = wav_files(i).name;
    
    % Extract the emotion label from the file name
    emotion_label = file_name(4); % Assuming the label is the sixth character of the file name
    emotion_name = ''; % Initialize emotion name
    
    % Map the emotion label to its corresponding name
    switch emotion_label
        case 'a'
            emotion_name = 'Angry';
        case 'd'
            emotion_name = 'Disgust';
        case 'f'
            emotion_name = 'Fearful';
        case 'h'
            emotion_name = 'Happy';
        case 'n'
            emotion_name = 'Neutral';
        case 's'
            if file_name(5) == 'a'
                emotion_name = 'Sad';
            elseif file_name(5) == 'u'
                emotion_name = 'Surprise';
            else
                emotion_name = 'Unknown'; % If the third character is neither 'a' nor 'u'
            end
        otherwise
            emotion_name = 'Unknown';
    end
    
    % Display the emotion name for each file
    fprintf('%s\n', emotion_name);

    % Construct the full path to the current WAV file
    current_file_path = fullfile(dataset_directory, file_name);

    % Load the audio file and sampling rate
    [audioIn, fs] = audioread(current_file_path);
    

    %========================================================================================================
    % Compute MFCC, Delta MFCC, DeltaDelta MFCC
    win = hann(WindowSize, 'periodic');
    S = stft(audioIn, 'Window', win, 'OverlapLength', Overlap, 'Centered', false);
    [coeffs,delta,deltaDelta] = mfcc(S, fs, 'LogEnergy', 'Ignore');
    % Compute the mean along the columns
    coeffs = mean(coeffs, 1);
    % Compute the mean of coeffs
    mean_coeffs = mean(coeffs);
    % Compute the standard deviation of coeffs
    std_coeffs = std(coeffs);
    % Compute the mean along the columns
    delta = mean(delta, 1);
    % Compute the mean of coeffs
    mean_delta = mean(delta);
    % Compute the standard deviation of delta
    std_delta = std(delta);
    % Compute the mean along the columns
    deltaDelta = mean(deltaDelta, 1);
    % Compute the mean of coeffs
    mean_deltaDelta = mean(deltaDelta);
    % Compute the standard deviation of deltaDelta
    std_deltaDelta = std(deltaDelta);
    %========================================================================================================


    %========================================================================================================
    % Compute MelSpectrogram
    S = melSpectrogram(audioIn, fs, 'Window',hann(WindowSize, 'periodic'), 'OverlapLength',Overlap).';
    % Compute the mean along the columns
    melspec = mean(S, 1);
    % Compute the mean of melspec
    mean_melspec = mean(melspec);
    % Compute the standard deviation of melspec
    std_melspec = std(melspec);
    %========================================================================================================


    %========================================================================================================
    % Compute SpectralFlux
    flux = spectralFlux(audioIn, fs, 'Window',hann(WindowSize,'periodic'), 'OverlapLength',Overlap);
    % Compute the mean along the columns
    specflux = mean(flux, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute SpectralSkewness
    skewness = spectralSkewness(audioIn, fs, 'Window',hann(WindowSize,'periodic'), 'OverlapLength', Overlap);
    skewness(isnan(skewness)) = 0;
    % Compute the mean along the columns
    specSkewness = mean(skewness, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute Spectral Slope
    slope = spectralSlope(audioIn, fs, 'Window', hann(WindowSize, 'periodic'), 'OverlapLength', Overlap);
    % Compute the mean along the columns
    slope = mean(slope, 1);
    %========================================================================================================
    

    %========================================================================================================
    % Compute Spectral Entropy
    entropy = spectralEntropy(audioIn, fs, 'Window', hann(WindowSize, 'periodic'), 'OverlapLength', Overlap);
    % Compute the mean along the columns
    entropy = mean(entropy, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute Spectral Rolloff
    rolloff = spectralRolloffPoint(audioIn, fs, 'Window', hann(WindowSize, 'periodic'), 'OverlapLength', Overlap);
    % Compute the mean along the columns
    rolloff = mean(rolloff, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute Chromagram
    % Calculate the Short-Time Fourier Transform (STFT)
    [S, F, T] = stft(audioIn, fs, 'Window', hamming(WindowSize), 'OverlapLength', Overlap, 'Centered', false);
    % Define the mapping from frequency bins to chroma
    % Assuming 12-note equal temperament
    chroma_mapping = mod(round(12*log2(F/440)*12), 12) + 1; % 12 chroma bins
    % Initialize matrix to store chroma features
    chroma_features = zeros(size(S,2), 12);
    % Compute chroma features
    for j = 1:12
        bin_indices = chroma_mapping == j;
        chroma_features(:, j) = sum(abs(S(bin_indices, :)), 1);
    end  
    chromagram = mean(chroma_features, 1);
    mean_chromagram = mean(chromagram);
    std_chromagram = std(chromagram);
    %========================================================================================================


    %========================================================================================================
    % Calculate number of windows and hop size
    num_samples = length(audioIn);
    numWindows = floor((num_samples - WindowSize) / Overlap) + 1;
    hopSize = WindowSize - Overlap;
    % Initialize variables to store LPC features for all windows
    all_lpc_coefficients = zeros(numWindows, order);
    % Iterate through each window
    for j = 1:numWindows
        % Extract the windowed audio segment
        startIndex = (j - 1) * hopSize + 1;
        endIndex = startIndex + WindowSize - 1;
        windowedAudio = audioIn(startIndex:endIndex);
        % Apply windowing (e.g., Hamming window)
        windowedAudio = windowedAudio .* hamming(WindowSize);
        % Perform LPC analysis on the windowed segment
        [a, ~] = lpc(windowedAudio, order);
        % Exclude the gain term (first coefficient)
        a = a(2:end);
        % Store LPC coefficients for this window
        all_lpc_coefficients(j, :) = a;
    end
    inf_indices = any(isinf(all_lpc_coefficients), 2);
    all_lpc_coefficients(inf_indices, :) = 0;
    nan_indices = any(isnan(all_lpc_coefficients), 2);
    all_lpc_coefficients(nan_indices, :) = 0;
    lpc_coeff = mean(all_lpc_coefficients);
    % Calculate mean and standard deviation of LPC coefficients across windows
    mean_lpc = mean(lpc_coeff);
    std_lpc = std(lpc_coeff);
    %========================================================================================================


    %========================================================================================================
    % Compute Zero Crossing Rate (ZCR)
    zcr_batch = zerocrossrate(audioIn, 'WindowLength', WindowSize, 'OverlapLength', Overlap);
    % Compute the mean along the columns
    zcr = mean(zcr_batch, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute Energy
    % Initialize matrices to store Energy features
    total_energy_features = zeros(length(wav_files), 1);
    mean_energy_features = zeros(length(wav_files), 1);
    std_energy_features = zeros(length(wav_files), 1);
    % Calculate the number of frames
    num_samples = length(audioIn);
    num_frames = floor((num_samples - WindowSize) / Overlap) + 1;
    % Initialize vectors to store frame energies
    frame_energy = zeros(1, num_frames);
    % Compute energy for each frame
    for frame = 1:num_frames
        % Determine the start and end indices of the current frame
        start_index = (frame - 1) * Overlap + 1;
        end_index = start_index + WindowSize - 1;
        % Extract the current frame
        frame_data = audioIn(start_index:end_index);
        % Calculate the energy of the current frame
        frame_energy(frame) = sum(frame_data.^2);
    end
    % Compute total energy, mean energy, and standard deviation of energy
    total_energy = sum(frame_energy);
    mean_energy = mean(frame_energy);
    std_energy = std(frame_energy);
    %========================================================================================================


    %========================================================================================================
    % Compute Pitch
    f0 = pitch(audioIn, fs, 'WindowLength', WindowSize, 'OverlapLength',Overlap);
    % Compute the mean along the columns
    f0 = mean(f0, 1);
    %========================================================================================================


    %========================================================================================================
    % Compute Intensity
    % Reference intensity (in watts per square meter)
    I0 = 1e-12;
    % Calculate the number of frames
    num_samples = length(audioIn);
    num_frames = floor((num_samples - WindowSize) / Overlap) + 1;
    % Initialize vectors to store intensity of each frame
    frame_intensity = zeros(1, num_frames);
    % Compute Intensity for each frame
    for frame = 1:num_frames
        % Determine the start and end indices of the current frame
        start_index = (frame - 1) * Overlap + 1;
        end_index = start_index + WindowSize - 1;
        % Extract the current frame
        frame_data = audioIn(start_index:end_index);
        % Calculate the Intensity of the current frame
        intensity_value = mean(frame_data.^2); % Square the amplitude and take the mean
        % Convert Intensity to dB
        intensity_dB = 10 * log10(intensity_value / I0);
        % Store Intensity for the current frame
        frame_intensity(frame) = intensity_dB;
    end
    inf_indices = any(isinf(frame_intensity), 2);
    frame_intensity(inf_indices, :) = 0;
    nan_indices = any(isnan(frame_intensity), 2);
    frame_intensity(nan_indices, :) = 0;
    intensity = mean(frame_intensity);
    %========================================================================================================


    %========================================================================================================
    % Compute HNR
    noOfFrames = floor((length(audioIn)-WindowSize)/Overlap)+1;
    hnr = zeros(noOfFrames, 1);
    for j = 1:noOfFrames
        startIdx = (j-1)*Overlap+1;
        endIdx = startIdx+WindowSize-1;
        xFrame = audioIn(startIdx:endIdx);
        [r, lag] = xcorr(xFrame);
        r = r(lag>=0);
        r = r/max(r);
        [pks, locs] = findpeaks(r);
        if isempty(locs)
            hnr(j) = 0;
        else
            hnr(j) = 10*log10(sum(pks(2:end).^2)/pks(1)^2);
        end
    end
    inf_indices = any(isinf(hnr), 2);
    hnr(inf_indices, :) = 0;
    nan_indices = any(isnan(hnr), 2);
    hnr(nan_indices, :) = 0;
    % Compute the mean along the columns
    mean_hnr = mean(hnr);
    %========================================================================================================


    %========================================================================================================
    % Compute RMS
    rms = zeros(1, length(audioIn));
    for j = 1:Overlap:length(audioIn)-WindowSize
        % Extract window
        window = audioIn(j:j+WindowSize-1);
        % Calculate RMS for the window
        rms(j:j+WindowSize-1) = sqrt(sum(window.^2) / WindowSize);
    end
    % Compute the mean along the columns
    rms = mean(rms);
    %========================================================================================================



    % Save all the features to the matrix
    all_statistics(i, :) = [coeffs, mean_coeffs, std_coeffs, delta, mean_delta, std_delta, ...
                            deltaDelta, mean_deltaDelta, std_deltaDelta, melspec, mean_melspec, std_melspec, ...
                            specflux, specSkewness, slope, entropy, rolloff, chromagram, mean_chromagram, std_chromagram, ...
                            lpc_coeff, mean_lpc, std_lpc, zcr, total_energy, mean_energy, std_energy, f0, intensity, mean_hnr, rms];
end

% Save the matrix to a CSV file
csvwrite('SAVEE_Features.csv', all_statistics);