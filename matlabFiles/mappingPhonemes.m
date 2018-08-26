

fields = fieldnames(phonemeSeqCells);

% phonemeSeqCells.(fields{10});

% name = strcat('data',int2str(num),'.txt');
% 
% data = [features output];
% 
% fid = fopen(name, 'a');
% fclose(fid);
% 
% header2(1:8) = {'DataRate','CameraRate','NumFrames','NumMarkers','Units','OrigDataRate','OrigDataStartFrame','OrigNumFrames'};
% 
% dlmwrite(name, data(1:end,:), '-append', 'delimiter', '\t');


for i = 1:numel(fields)
    sequence = [];
    frameNums = [];
    sequence = [sequence phonemeSeqCells.(fields{i})];
    frameNums = [frameNums 1:length(phonemeSeqCells.(fields{i}))];
    name = strcat('recordingVolunteerFiles/', fields{i}, '.txt');
    fid = fopen(name, 'w');
    fprintf(fid, '%u\t', frameNums(1:end-1));
    fprintf(fid, '%u\n', frameNums(end));
    fprintf(fid, '%s\t', sequence(1:end-1));
    fprintf(fid, '%s\n', sequence(end));
    fclose(fid);

end



% 
% dlmwrite(name, frameNums(1:end), '-append', 'delimiter', '\t');
% dlmwrite(name, 