% Clean Stanford FingerFlex ECoG using training-only channel variance.
%
% Input: CORTEG_STANFORD_RAW_ROOT/<subject>/<subject>_fingerflex.mat,
% containing data (time x raw channels) and flex (time x five fingers).
% Semantic settings: <repo>/configs/stanford_preprocessing.json.
% Output: <repo>/preprocessed_data/Stanford/<subject>.mat containing
% data (retained channels x time), flex, fs, retained channelName,
% badChannelName, channelVariance, medianChannelVariance, trainSampleCount,
% and the two variance multipliers. The output root also contains
% resolved_config.json. The raw root is read only.
%
% Requires MATLAB and FieldTrip fieldtrip-20230926.

close all
clear

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
configPath = fullfile(repoRoot, 'configs', ...
    'stanford_preprocessing.json');
configSource = jsondecode(fileread(configPath));
config = mergeConfig(configSource.defaults, configSource.overrides);
validateConfig(config);
rawRoot = getenv('CORTEG_STANFORD_RAW_ROOT');
outputDir = fullfile(repoRoot, 'preprocessed_data', 'Stanford');
provenancePath = fullfile(outputDir, 'resolved_config.json');

if isempty(rawRoot) || ~isfolder(rawRoot)
    error(['Set CORTEG_STANFORD_RAW_ROOT to the folder containing ' ...
        '<subject>/<subject>_fingerflex.mat.']);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end
subjectCodes = {'bp', 'cc', 'ht', 'jc', 'jp', ...
    'mv', 'wc', 'wm', 'zt'};
if ~isfile(provenancePath)
    for iSubject = 1:numel(subjectCodes)
        priorPath = fullfile(outputDir, ...
            [subjectCodes{iSubject} '.mat']);
        if isfile(priorPath)
            error(['Cleaned data exist without resolved_config.json; ' ...
                'refusing to adopt %s'], priorPath);
        end
    end
end
if isfile(provenancePath)
    previousConfig = jsondecode(fileread(provenancePath));
    if ~isequaln(previousConfig, config)
        error(['Existing cleaned output uses different semantic ' ...
            'configuration: %s'], provenancePath);
    end
else
    provenanceFile = fopen(provenancePath, 'w');
    if provenanceFile < 0
        error('Cannot write preprocessing provenance: %s', ...
            provenancePath);
    end
    fprintf(provenanceFile, '%s\n', jsonencode(config));
    fclose(provenanceFile);
end

fs = 1000; % Dataset's documented sample rate; raw mat lacks fs.
for iSubject = 1:numel(subjectCodes)
    subject = subjectCodes{iSubject};
    rawPath = fullfile(rawRoot, subject, ...
        [subject '_fingerflex.mat']);
    outputPath = fullfile(outputDir, [subject '.mat']);
    if ~isfile(rawPath)
        error('Missing raw Stanford recording: %s', rawPath);
    end
    if isfile(outputPath)
        saved = load(outputPath, ...
            'varianceHighMultiplier', 'varianceLowMultiplier');
        if ~isfield(saved, 'varianceHighMultiplier') || ...
                ~isfield(saved, 'varianceLowMultiplier') || ...
                saved.varianceHighMultiplier ~= ...
                config.bad_channel.high_variance_multiplier || ...
                saved.varianceLowMultiplier ~= ...
                config.bad_channel.low_variance_multiplier
            error('Existing cleaned recording is incompatible: %s', ...
                outputPath);
        end
        fprintf('%s: reusing existing cleaned recording %s\n', ...
            subject, outputPath);
        continue
    end
    recording = load(rawPath, 'data', 'flex');
    if ~isfield(recording, 'data') || ~isfield(recording, 'flex')
        error('Expected data and flex in raw recording: %s', rawPath);
    end
    data = recording.data' * config.ecog_scale;
    flex = recording.flex;
    clear recording
    if size(data, 2) ~= size(flex, 1)
        error('%s: ECoG and flex sample counts differ.', subject);
    end
    if size(data, 1) < 2 || size(data, 2) < 2
        error('%s: expected at least two channels and two samples.', ...
            subject);
    end
    if any(~isfinite(data(:)))
        error('%s: raw ECoG contains NaN or infinity.', subject);
    end

    for iBand = 1:size(config.notch_bands_hz, 1)
        data = ft_preproc_bandstopfilter( ...
            data, fs, config.notch_bands_hz(iBand, :), ...
            config.notch_filter_order, ...
            config.notch_filter_family, ...
            config.notch_filter_direction);
    end

    if floor(size(data, 2) / fs) >= config.split.duration_cutoff_seconds
        trainSampleCount = config.split.at_or_above_cutoff.train_seconds * fs;
    else
        trainSampleCount = floor( ...
            size(data, 2) / config.split.below_cutoff.total_parts) ...
            * config.split.below_cutoff.train_parts;
    end
    if trainSampleCount < 2 || trainSampleCount >= size(data, 2)
        error('%s: invalid training split at sample %d of %d.', ...
            subject, trainSampleCount, size(data, 2));
    end

    channelName = arrayfun( ...
        @(index) sprintf('ch%d', index), 1:size(data, 1), ...
        'UniformOutput', false)';
    channelVariance = var( ...
        data(:, 1:trainSampleCount), ...
        config.bad_channel.variance_normalization, 2);
    if any(~isfinite(channelVariance))
        error('%s: non-finite training-channel variance.', subject);
    end
    medianChannelVariance = median(channelVariance);
    if medianChannelVariance <= 0
        error('%s: median training-channel variance must be positive.', ...
            subject);
    end
    varianceHighMultiplier = ...
        config.bad_channel.high_variance_multiplier;
    varianceLowMultiplier = ...
        config.bad_channel.low_variance_multiplier;
    badMask = channelVariance > ...
        varianceHighMultiplier * medianChannelVariance | ...
        channelVariance < ...
        varianceLowMultiplier * medianChannelVariance;
    if sum(~badMask) < 2
        error('%s: variance rule retained fewer than two channels.', ...
            subject);
    end
    badChannelName = channelName(badMask);
    channelName = channelName(~badMask);
    data = data(~badMask, :);
    data = ft_preproc_rereference( ...
        data, config.reference.channels, config.reference.method);

    save(outputPath, 'data', 'flex', 'fs', 'channelName', ...
        'badChannelName', 'channelVariance', ...
        'medianChannelVariance', 'trainSampleCount', ...
        'varianceHighMultiplier', 'varianceLowMultiplier');
    fprintf('%s: removed %d of %d channels (%s). Saved %s\n', ...
        subject, numel(badChannelName), numel(badMask), ...
        strjoin(badChannelName', ', '), outputPath);
end

function resolved = mergeConfig(defaults, overrides)
    % Recursively resolve JSON overrides without dropping default keys.
    resolved = defaults;
    keys = fieldnames(overrides);
    for iKey = 1:numel(keys)
        key = keys{iKey};
        if ~isfield(defaults, key)
            error('Unknown preprocessing override: %s', key);
        end
        if isstruct(defaults.(key)) && isstruct(overrides.(key))
            resolved.(key) = mergeConfig( ...
                defaults.(key), overrides.(key));
        else
            resolved.(key) = overrides.(key);
        end
    end
end

function validateConfig(config)
    % Reject invalid settings before touching recordings.
    low = config.bad_channel.low_variance_multiplier;
    high = config.bad_channel.high_variance_multiplier;
    if ~isscalar(low) || ~isscalar(high) || ...
            ~isfinite(low) || ~isfinite(high) || ...
            low <= 0 || low >= 1 || high <= 1
        error('Expected 0 < low variance multiplier < 1 < high.');
    end
    bands = config.notch_bands_hz;
    scale = config.ecog_scale;
    if ~isscalar(scale) || ~isfinite(scale) || scale <= 0
        error('ecog_scale must be finite and positive.');
    end
    cutoff = config.split.duration_cutoff_seconds;
    trainSeconds = config.split.at_or_above_cutoff.train_seconds;
    if ~isscalar(cutoff) || ~isscalar(trainSeconds) || ...
            ~isfinite(cutoff) || ~isfinite(trainSeconds) || ...
            cutoff <= 0 || trainSeconds <= 0 || ...
            trainSeconds >= cutoff
        error('Expected 0 < split.at_or_above_cutoff.train_seconds < cutoff.');
    end
    denominator = config.split.below_cutoff.total_parts;
    numerator = config.split.below_cutoff.train_parts;
    if ~isscalar(denominator) || ~isscalar(numerator) || ...
            ~isfinite(denominator) || ~isfinite(numerator) || ...
            denominator ~= floor(denominator) || ...
            numerator ~= floor(numerator) || ...
            numerator < 1 || numerator >= denominator
        error('Expected 1 <= split.below_cutoff.train_parts < total_parts.');
    end
    normalization = config.bad_channel.variance_normalization;
    if ~ismember(normalization, [0 1])
        error('variance_normalization must be 0 or 1.');
    end
    if ~isnumeric(bands) || size(bands, 2) ~= 2 || ...
            isempty(bands) || any(~isfinite(bands(:)))
        error('notch_bands_hz must be a nonempty finite N-by-2 array.');
    end
end
